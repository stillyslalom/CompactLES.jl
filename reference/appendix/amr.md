# Calibration appendix: Refinement

This file holds the sections of the measurement record on the refined levels: transfer,
coupling, conservation, regridding, tiling and cost.
[CALIBRATION_APPENDIX.md](../CALIBRATION_APPENDIX.md) indexes the record and states its
conventions.

## AMR

Patch AMR, the level hierarchy and the device backend. The mechanism is in
`reference/AMR_GPU.md`; the numbers are here.

### bench/interfaceconservation.jl: composite conservation budgets

`mpiexec -n 8 julia --project=. -t 1 bench/interfaceconservation.jl N=96 ny=24
tfinal=50.26548245743669 moving_tfinal=50.26548245743669 samples=16 check=true`,
8.5 minutes on the i9-12900K.

The case is periodic passive-species shear: two thermodynamically identical
ideal species at uniform density and pressure, a uniform stream with a weak
sinusoidal transverse velocity, and the composition a periodic pair of
perturbed tanh sheets. There is no physical transport, the artificial
properties are off except the mass-fraction bound, and validity is
permissive (below). The layouts are the uniform grid, two same-level patches,
a two-level and a three-level nest each with global steps and subcycled, and
the two-level nest regridded sixteen times under species-gradient tagging.

```text
domain           [0, 8π) × [0, 2π), periodic
grid             96 × 24, Float64
flow             u = 1, v = 0.08 sin(x/4)
sheet width      0.22 (fixed layouts), 0.18 (moving refinement)
scheme           C6, the default C8 filter every step, CFL 0.45
duration         two domain transits, t = 16π
```

The budgets were declared before the measurement. The momentum budget is
scaled by M₀c₀, which gives the initially zero transverse components a scale.

| quantity | budget |
|---|---|
| each species mass, total mass, total energy | 1e-3 of the initial value |
| each momentum component | 1e-3 of M₀c₀ |
| each layout's initial sampling offset from the uniform run | the same |
| molecular mixing | 0.01 |
| mix width | 1% of the domain length |

The conserved snapshot is the quadrature the coarse-fine
coupling conserves (`volume_integral`'s composite form) with shared planes
counted once, and each layout's drift is taken from its own
initial integral, sampled sixteen times so that a cancelling excursion is not
hidden by the endpoint. The regrid ledger records the jump across each
transfer separately from the evolution between transfers.

**Fixed layouts.** The drift is an exchange between the two species: total
mass, momentum and energy hold to 4e-11 in every layout, and the two species
masses move by equal and opposite amounts. The table gives the sampled
maximum of the normalized drift over two transits and the largest
mass-fraction excursion outside [0, 1] seen at any sample.

| layout | root steps | max drift | species excursion |
|---|---|---|---|
| uniform | 1184 | 5e-14 (round-off) | 0 |
| two same-level patches | 1187 | 5.5e-10 | 3.4e-4 |
| two levels, global step | 3532 | 7.1e-7 | 0 |
| two levels, subcycled | 1184 | 4.9e-7 | 0 |
| three levels, global step | 10568 | 8.5e-7 | 3.6e-4 |
| three levels, subcycled | 1184 | 4.8e-7 | 0 |

The refined layouts also agree with the uniform run, each comparison an order
of magnitude or more inside its budget:

| comparison with the uniform run | largest | budget |
|---|---|---|
| initial sampling offset, two and three levels | 3.0e-10 | |
| mix width, over the history | 1.6e-3 | 0.25 |
| molecular mixing, over the history | 8.2e-4 | 0.01 |

**Moving refinement.** The evolution between transfers is the larger term of
the drift and grows steadily over the run; the regrid jumps are the smaller
one. Sixteen regrids over two transits, global step and subcycled:

| quantity | global step | subcycled |
|---|---|---|
| per-regrid jump, median | 1.4e-7 | 3.2e-7 |
| sum of the absolute jumps | 3.8e-6 | 5.8e-6 |
| sampled max drift, transfers included | 1.7e-5 | 1.3e-5 |
| sampled max drift, transfers subtracted | 1.5e-5 | 8.9e-6 |
| final drift | 1.6e-5 | 1.1e-5 |
| species excursion | 7.1e-4 | 5.8e-4 |

The budget compares the absolute sum of the jumps, which at this spacing
would reach it after some three thousand regrids. Regridding a three-level
hierarchy is unsupported and is not measured.

**Why the runs are permissive, and the bound is on.** The mass-fraction
undershoot that ends some of these runs under strict validity is independent
of conservation: the drift up to the point of failure was inside the budget
in every failed run, so the instrument reports the excursion beside the drift
instead of stopping on it. With the artificial properties off entirely,
strict validity ended the runs below on mass fractions past the 1e-4 dead
band.

| run, artificial properties off, strict validity | ended at |
|---|---|
| three levels, global step | t = 37.7 |
| moving refinement, long run | t = 25.1 |
| uniform, unfiltered | t = 28.3 |

A per-step probe of the three-level run shows no growth anywhere else:
velocity, pressure and density stay at their initial values to four digits,
and the step size is steady. The undershoot sits on the root and first-level
nodes along the edges of the nested box and is advected with the flow. It
grows from the size of the dead band at t = 34 to three times it at t = 38. The
mass-fraction bound at its default slows it but does not hold it below the
dead band, since it acts in proportion to the excursion. The same layout
subcycled shows no excursion at all.

**Decision.** Every layout is inside its conservation budget, the fixed
layouts by an order of magnitude, so smooth flow takes no surface-flux
correction; the correction of
[bench/archive/reflux.jl](#bencharchiverefluxjl-the-conservative-coarse-fine-coupling) is
gated to lines beside an unresolved density feature, and the density of these
cases is uniform. The root-edge
mass-fraction undershoot under global stepping is an imposed-shell accuracy
question for N11, not a conservation one. No turbulent, variable-density,
shock or hardware-GPU claim follows from these passive-species cases.

**Attribution.** On a refined layout the right-hand side's share of the
drift is the coarse-fine flux mismatch at the junction, and the rest is the
filter's; each level's right-hand side less the fluxes through its faces
conserves to round-off.
`ledger=true` books each run's drift to the mechanism that made it
(DESIGN.md, [attributing the conserved
budgets](../DESIGN.md#attributing-the-conserved-budgets)). The table is species 1
on its initial mass, two levels subcycled at the resolution above, over two
transits at eight ranks in 18.7 minutes. The column "at t = 15π" is the
sample of largest drift. Total mass, momentum and energy stay at 4e-11 or
below in every row, so the species exchange is the whole drift. Rows absent
from a run (`shell`, `restrict`, `same_level` on a refined layout,
`unattributed`) were zero: the shell and the restriction write only nodes the
quadrature does not count.

| piece | at t = 15π | at t = 16π |
|---|---|---|
| right-hand side, both levels | −3.68e-7 | −4.03e-8 |
| of which: root, less its face fluxes | −4e-15 | −6e-15 |
| of which: level 1, less its face fluxes | +7e-16 | −2e-15 |
| of which: coarse–fine flux mismatch | −3.68e-7 | −4.03e-8 |
| filter | −1.17e-7 | −9.79e-8 |
| total drift | −4.85e-7 | −1.38e-7 |
| telescoping residual | 1e-15 | 1e-15 |
| right-hand side less its stage integral | 6e-15 | 7e-15 |

The mismatch is each side's face flux at the junction's half node, the
child's with the rate of Ω, integrated through the stage recurrence: the
quantity the coarse-fine correction removes on a line it flags, and here
it flags none. Each level's right-hand side less the fluxes through its
faces is round-off, so the right-hand side's whole drift is the junction's.
The same split holds under the global step and at three levels; the
columns are the sample of largest drift and the end of the run.

| species 1 | mismatch, largest | mismatch, end | filter, largest | filter, end |
|---|---|---|---|---|
| two levels, global step (largest at t = 15π) | −5.15e-7 | −9.56e-8 | −1.91e-7 | −2.37e-7 |
| three levels, global step | −5.94e-7 | −1.79e-7 | | |
| three levels, subcycled | −3.63e-7 | −4.05e-8 | | |

Without a coarse-fine junction the filter's share is round-off, and the drift
of the same-level patches is the right-hand side's. Under moving refinement the right-hand side's part, all of it mismatch under the
global step, outweighs the sum of the regrid jumps.

| species 1 | right-hand side | filter | same-level averaging | regrid jumps, summed |
|---|---|---|---|---|
| uniform | 3e-15 | 5e-14 | | |
| two same-level patches | 5.5e-10 | 2.7e-14 | 2e-15 | |
| moving refinement, global step | +1.38e-5 | +8.9e-7 | | +1.0e-6 |
| moving refinement, subcycled | +5.9e-6 | +3.0e-6 | | +1.65e-6 |

The drift is a truncation error of the coupling at the junction, converging
at high order once Ω is counted: doubling the resolution of the two-level
subcycled run reduces its largest drift over t ≤ 5π by a factor of 93.

| two levels, subcycled, t ≤ 5π | 96 × 24 | 192 × 48 |
|---|---|---|
| largest sampled drift | 2.05e-7 | 2.19e-9 |
| mismatch at t = 5π | 2.08e-7 | 8.7e-11 |

**Decision.** Smooth flow takes no conservative correction. The attributed
non-conservation is the junction's flux mismatch, more than three orders of
magnitude inside the budget at this resolution and converging at high order,
and a correction applied on every line of a 2-D face costs the interface its
order ([bench/archive/reflux.jl](#bencharchiverefluxjl-the-conservative-coarse-fine-coupling)).

### bench/archive/reflux.jl: the conservative coarse-fine coupling

`julia --project=. -t 4 bench/archive/reflux.jl`, serial, Float64, C6 and the default
filter; the cases are described in the script's header. The uncorrected rows
ran with the correction off (`conservative = false`).

**Sod.** Across a shock that crosses both faces of the box, the correction
reduces the composite mass and energy change by about three to four orders
of magnitude, under either stepping and with the parent mask off or on. The
box is over [0.6, 0.8]. The table gives the relative change at t = 0.2 in
the conserved quadrature (`_conserved_budget`):

| stepping | parent mask | uncorrected mass | corrected mass | uncorrected energy | corrected energy |
|---|---|---|---|---|---|
| global | off | +8.51e-5 | +6.65e-8 | +6.46e-5 | +3.90e-8 |
| global | on | −5.23e-4 | −8.93e-7 | −7.71e-4 | −6.38e-7 |
| subcycled | off | +9.12e-5 | +3.10e-8 | +4.72e-5 | +5.90e-8 |
| subcycled | on | −7.48e-4 | +3.72e-8 | −1.05e-3 | +6.58e-8 |

**Entropy wave.** A correction applied on every line costs a smooth
two-dimensional interface its order, and the gate keeps it off smooth data.
Largest density error over the level's nodes at t = 0.3:

| coupling | N = 24 | N = 48 | N = 96 | orders |
|---|---|---|---|---|
| uncorrected | 1.929e-8 | 3.006e-10 | 4.915e-12 | 6.00, 5.93 |
| gated (default) | 1.929e-8 | 3.006e-10 | 4.915e-12 | 6.00, 5.93 |
| every line corrected | 7.642e-4 | 2.129e-4 | 1.245e-4 | 1.84, 0.77 |

The gate never fires on the wave, and the gated rows are the uncorrected ones
bit for bit. Corrected on every line, each line takes the difference of the
two grids' tangential quadratures of its face flux, of order H³ per line, and
the level's error falls to second order and below. On a 1-D line, where there
is no tangential quadrature, the ungated correction keeps the interface's
order; without Ω, the correction of the box rule beside the face, it lowers
it to 2.

**Noh crossing.** When a shock leaves the tile, the correction holds the
composite mass to the uniform run's own drift, where the uncorrected coupling
loses about a thousandth of it. The tile over 43 root nodes passes its shock
out near t = 0.5. The table gives the mass pieces of the budget ledger over
the run to t = 0.6, and the composite mass less the inflow through the outer
boundary, as fractions of the final mass:

| coupling | right-hand sides | filter | correction | inflow | composite less inflow |
|---|---|---|---|---|---|
| uncorrected | 0.778337 | +1.51e-4 | 0 | 0.780000 | −1.18e-3 |
| corrected | 0.776899 | +1.13e-3 | +1.97e-3 | 0.780000 | −3.65e-7 |

The uniform run at the level's spacing drifts by −4.45e-7 against the exact
mass. The crossing leaves a static density crater beside the face, ρ ≈ 8.9
against a plateau near 15 on the child, three child nodes wide, which is
present uncorrected as well. On data that rough Ω is not small, and the
cells alone misread the corrected run's mass by its share; the conserved
quadrature, which `volume_integral` takes, reads the composite of the table
above. Ω grows from t ≈ 0.45 as the shock reaches the face.

| corrected run, of the final mass | t = 0.2 | t = 0.6 |
|---|---|---|
| Ω | −4e-7 | −3.96e-4 |
| the cells alone | | +3.96e-4 |

**A moving converging shock.** A gate that tests the covered nodes corrects a
moving shock's junction in nearly every step, and the correction lands in the
gas at rest ahead of the shock; testing the uncovered nodes alone, the
default, keeps the masked level's error ahead of the shock at the fine
grid's. The
shock problem of `bench/archive/movinglevel.jl` (256 root nodes, a subcycled box
placed by the density sensor four root nodes ahead of the shock) keeps the
shock inside the box's inner face and within `GATE_REACH` of it. The moving
shock column is the level's largest |ρ − 1| beyond 0.02 inside the front,
with variants of the gate; the Sod and Noh columns summarize the tables above.

| gate | mask | moving shock | Sod mass, largest | Noh composite less inflow |
|---|---|---|---|---|
| uniform fine grid | | 2.91e-4 | | |
| none (`conservative = false`) | off | 1.08e-2 | 9.1e-5 | −1.18e-3 |
| none | on | 3.38e-4 | 7.5e-4 | |
| parent nodes either side of the face | off | 1.92e-2 | 6.7e-8 | −3.65e-7 |
| the same | on | 3.56e-3 | 8.9e-7 | |
| covered nodes only | on | 3.56e-3 | | |
| uncovered nodes only | off | 2.61e-2 | 2.1e-6 | +4.92e-6 |
| uncovered nodes only (default) | on | 3.11e-4 | 1.2e-6 | −1.72e-5 |
| either side within 2 nodes | on | 4.80e-3 | | |
| either side, correction on the face node | on | 3.00e-3 | | |

Tested either side of the face, the gate fires on the restricted shock the
covered nodes hold. Testing only the uncovered nodes, the default, brings
the masked level to 1.07 times the fine grid's, where testing either side
leaves it at 12, and loosens the Sod and Noh conservation
by one to two orders, still two orders below the uncorrected coupling's.
The default's row was re-measured with the parent's sensors dropped at its
overwritten nodes (`MASK_CHILD_SENSORS`); the other rows predate that mask.

**Where the default's drift comes from.** With every line corrected in every
step the Sod tube conserves mass to the unrefined run's round-off with the mask
off or on. The default's drift there is the
mismatch of the steps the gate leaves uncorrected, which the registers
discard, and the mask makes the uncorrected mismatch about six times larger.
The Noh crossing without the mask behaves the same way. With the mask a
remainder survives the correction of every line, opposite in sign to the
gate's share, so the default's composite on the crossing is the sum of two
pieces that partly cancel. The right-hand side's change less its registered
face fluxes is the same with the mask off and on, which places the remainder
in the filter passes. Planar Noh crossings, with the level at the wall and
off it, show no such remainder: there the mask moves the filter and
correction pieces together by about 1e-7. The table gives the composite mass
change of the two-level tube of `test/level_tests.jl` at t = 0.2, and of the
Noh crossing less its inflow as a fraction of the exact mass, from the budget
ledger (a scratch driver, not kept; `b920d02`, Julia 1.13.3).

| case | mask | uncorrected | gated (default) | every line corrected |
|---|---|---|---|---|
| Sod tube | off | +8.52e-5 | +2.09e-6 | −4.9e-11 |
| Sod tube | on | −5.23e-4 | −6.98e-6 | −1.0e-10 |
| Noh crossing | off | −1.18e-3 | +4.92e-6 | −2.5e-7 |
| Noh crossing | on | −1.72e-3 | +8.38e-6 | −7.37e-6 |
| unrefined tube | | −5.5e-11 | | |
| uniform Noh at the level's spacing | | −4.45e-7 | | |

**Rejected placements.** A correction added to the parent's face node, which
the shell copies to the child, fed back through the child's next step and
grew without bound in the cold radial inflow; the node one out from the face
does not. A gate window over the whole box flagged the pre-shock flow, whose
density ρ = 1 + t/r varies steeply near the axis, and the correction it
applied there polluted the inflow; the window of `GATE_REACH` = 8 parent
nodes either side of the face does not reach it. Without the positivity
guard the Noh crossing reached a negative density at t = 0.459, and without
the filter passes' junction fluxes in the register the corrected composite
gained 4.9e-3 of its mass.

### bench/archive/interfacesensor.jl: the sensors and the filter at an interface

`julia --project=. -t 16 bench/archive/interfacesensor.jl` (`probe`, `crossing`,
`filter` and `undershoot`; the toggle `SENSOR_INTERFACE_GHOSTS` selects
the detector's interface taps).

**The operator probe.** The detector's ghost taps carry a coarse-fine face's
interpolation error at the shell's order and reproduce a same-level face
exactly, where clamped taps put a first-order error in the sensor. Of the
filter's interface rows, the one-sided rows are closest to the uniform
filter at a coarse-fine face and the extended rows at a same-level face. The
probe is one application of each operator on a smooth field at rest,

```text
ρ = 1 + 0.3 sin(2π(x + 1/7)) + 0.05 sin(10π(x + 1/7)),  p = 1,  u = 0  on [0, 1)
```

against the uniform periodic grid of the same spacing: the level-1 patch of
a two-level nest at h/3, its shell imposed from the root's own
initialization, and the two patches of a same-level pair. The entry is
|interface − uniform| over the input amplitude at the first node in from
the face; the orders are log₂ of successive ratios. The detector runs at
weight power zero on the internal energy.

| operator, face | N = 48 | 96 | 192 | orders |
|---|---|---|---|---|
| δ⁴, ghost taps, shell | 2.2e-5 | 3.3e-7 | 5.0e-9 | 6.06 / 6.07 |
| δ⁴, clamped taps, shell | 2.3e-2 | 1.2e-2 | 5.8e-3 | 1.00 / 1.00 |
| δ⁴, ghost taps, same level | 0 | 0 | 0 | bitwise |
| δ⁴, clamped taps, same level | 2.8e-2 | 1.6e-2 | 8.4e-3 | 0.78 / 0.96 |
| `:gaussian` smoother, shell | 5.9e-3 | 3.0e-3 | 1.5e-3 | 0.95 / 0.98 |
| `:compact` smoother, shell | 9.4e-11 | 3.7e-13 | 1.5e-15 | 7.98 / 7.94 |
| filter, extended rows, shell, node 2 | 1.4e-7 | 2.0e-9 | 3.0e-11 | 6.08 / 6.08 |
| filter, onesided rows, shell, node 2 | 6.5e-12 | 3.6e-13 | 1.6e-15 | 4.17 / 7.78 |
| filter, cascade rows, shell, node 2 | 3.2e-5 | 9.6e-6 | 2.6e-6 | 1.75 / 1.90 |
| filter, extended rows, same level | 2.4e-7 | 9.7e-10 | 3.8e-12 | 7.96 / 7.99 |
| filter, onesided rows, same level, node 2 | 7.0e-7 | 3.6e-9 | 2.4e-11 | 7.59 / 7.23 |
| filter, cascade rows, same level, node 2 | 3.5e-4 | 1.1e-4 | 2.8e-5 | 1.73 / 1.95 |

The clamp's error is confined to the two nodes it touches and is exactly
zero beyond them; the shell's interpolation error, which the ghost taps
carry, is of order 6. The `:gaussian` smoother's
closed-edge rows add a first-order error over four nodes where the
`:compact` smoother stays at its own order. The filter's identity row
leaves node 1 at round-off under every row set. At a coarse-fine face the
extended rows read the imposed ghosts and so carry the interpolation
error into nodes 2 to 6, three decades above the filter's own change on
this field; the one-sided rows read nothing and stay at round-off. At a
same-level face the ghosts are exact neighbour data and the ordering
reverses: the extended rows track the uniform filter at order 8 and the
closed rows depart from it. The cascade rows are the worst set at both
faces.

**The Sod crossing.** A Sod shock crosses a refined box under every
configuration of the main table below with no inadmissible point counted on
any step, and the
detector's tap policy moves nothing past the third digit on a
single-species shock. The case:

```text
root grid      N = 201 on [0, 1] between slip walls, CFL 0.4
level 1        a box over [0.6, 0.8]; the shock enters through the low face
               near t = 0.057 and leaves through the high face near t = 0.17
run            to t = 0.2, the default artificial properties and filter
base row       C6, subcycled, one tile, filter_cfl = 0.35, the default rows,
               ghost taps, interface_flux = :closure
```

Each other row changes one setting. Package at `b4388f9`, Julia 1.13.3, from
`julia --project=. -t 1 bench/archive/interfacesensor.jl crossing`, whose
transcript is `bench/archive/results/interfacesensor.txt`. The noise is the momentum
ahead of the shock at t = 0.1 and x > 0.85, the level suite's gate quantity,
and ρ_min and p_min are the minimum density and pressure over every patch on
any step. The shell number is the artificial diffusivity number
((μ\* + β\*)/ρ + κ\*/(ρc_p) + max D\*)/(ch) over the four fine nodes nearest
each face, at the times of its two columns, the first before the shock
arrives; the shell error is the density error against the uniform run at the
fine spacing over those nodes.

| row | noise | ρ_min | p_min | shell number, t = 0.03 | t = 0.1 | shell error, t = 0.2 |
|---|---|---|---|---|---|---|
| base | 5.1e-12 | 0.061 | 0.040 | 7.7e-5 | 2.4e-2 | 9.8e-3 |
| clamped taps | 5.1e-12 | 0.061 | 0.040 | 7.6e-5 | 2.4e-2 | 9.8e-3 |
| C10 | 1.5e-10 | 0.067 | 0.043 | 2.0e-4 | 3.6e-2 | 1.2e-2 |
| global step | 6.7e-12 | 0.062 | 0.038 | 1.6e-4 | 1.9e-2 | 1.0e-2 |
| tile 8 | 5.1e-12 | 0.061 | 0.040 | 7.7e-5 | 4.5e-1 | 7.5e-2 |
| three levels | 5.0e-12 | 0.061 | 0.039 | 7.3e-5 | 3.4e-2 | 9.7e-3 |
| `filter_cfl = 0` | 7.4e-12 | 0.054 | 0.033 | 1.7e-4 | 2.6e-2 | 9.1e-3 |
| `:cascade4` rows | 4.9e-12 | 0.061 | 0.040 | 9.7e-5 | 2.3e-2 | 8.4e-3 |
| `:brady_livescu` rows | 5.0e-12 | 0.061 | 0.040 | 4.6e-5 | 5.8e-2 | 7.8e-3 |
| artificial properties off | 2.6e-12 | 0.090 | 0.054 | 0 | 0 | 3.3e-3 |
| three same-level patches | 2.6e-10 | 0.062 | 0.041 | 6.9e-6 | 4.3e-1 | 7.5e-2 |
| two species, ghost taps | 5.3e-12 | 0.084 | 0.036 | 2.1e-4 | 3.0e-2 | 9.3e-3 |
| two species, clamped taps | 5.3e-12 | 0.084 | 0.036 | 2.0e-4 | 1.0e-1 | 9.3e-3 |

The minimum density is the step data's start-up transient at step 3, and
the uniform run reads the same 0.062 at both spacings. Under the tiled
level and the three-patch layout the shell window includes the tile and
patch faces the shock is crossing at t = 0.1, so those two rows read the
shock itself. The tap policy has no visible effect on a single-species
shock, whose own δ⁴ signal is orders above the clamp's error. On the
two-species contact the clamped taps raise the shell number during the
crossing 3.5-fold and move the shell error in the third digit. The contact
carries the composition through both faces with a mass-fraction excursion
over the shell about six orders below the excursion elsewhere:

| two-species contact, mass-fraction excursion | over the shell | elsewhere |
|---|---|---|
| ghost taps | 2.7e-10 | 1.4e-4 |
| clamped taps | 2.1e-10 | 1.5e-4 |

The `:brady_livescu` rows put 2.4 times the base row's sensor on the shell
during the crossing and complete it; no closure candidate fails this gate
and none is promoted by it.

Under `interface_flux = :closure` a discontinuity initialized on a
same-level plane is the one failure. The extended gradient rows read the
jump through the ghosts while the divergence keeps its one-sided rows, and
the mismatch on a step one node wide fails the run. A captured shock
arriving from the interior is several nodes wide and crosses either kind of
face without incident, as every other row shows, and the default ghost
fluxes complete the run that fails
([ghost fluxes](#benchboundaryorderjl-gflux-the-divergence-through-interface-ends-from-ghost-fluxes)).
With two patches the shared plane is node 101, the Sod diaphragm:

| Sod diaphragm on a same-level plane, closure rows | outcome |
|---|---|
| base | negative density at step 2, ρ = −50 at the plane |
| artificial properties off | the same failure at step 10 |
| filter off | the same failure at step 2 |
| one-sided interface rows | completes |
| diaphragm one node to the right of the plane | completes |
| diaphragm one node to the left of the plane | fails |
| diaphragm adjacent to the plane on a 200-node grid | fails |
| a coarse-fine box face on the diaphragm | completes |

**The filter's change at the shell.** Under the default extended rows the
filter acts three to five times harder on the two nodes beside a coarse-fine
face than in the fine interior, and fifteen times harder in two dimensions,
where the transverse pass also moves the plane node itself by its interior
level before the shell is re-imposed. The one-sided rows filter the whole
fine interior five to ten times harder, since their closed rows make content
the extended rows do not, and the cascade rows are four decades worse on the
smooth wave. The table gives the density change of each filter pass, its
maximum over the run at each distance from a coarse-fine face relative to
its maximum over the fine interior, global stepping.

```text
entropy wave   ρ = 1 + 0.1 sin(2πx), u = 1, p = 1, root N = 96, to t = 0.5
Sod crossing   the base case above, to t = 0.2
2-D wave       the wave on (96, 24, 1) with a two-dimensional box; the plane
               column is the transverse pass
```

| case, rows | interior | plane | node 2 | node 3 | node 4 | node 6 |
|---|---|---|---|---|---|---|
| wave, extended | 2.8e-11 | 8e-6 | 2.8 | 3.9 | 3.9 | 2.0 |
| wave, onesided | 3.1e-10 | 0 | 1.4 | 2.2 | 2.5 | 1.7 |
| wave, cascade | 4.3e-7 | 0 | 3.3 | 2.1 | 2.0 | 1.6 |
| Sod, extended | 2.8e-4 | 2e-13 | 5.5 | 5.3 | 3.1 | 1.2 |
| Sod, onesided | 1.6e-3 | 4e-14 | 1.1 | 1.7 | 1.9 | 1.5 |
| Sod, cascade | 3.1e-4 | 2e-13 | 8.8 | 7.7 | 4.1 | 2.0 |
| 2-D wave, extended | 8.4e-10 | 1.4 | 14 | 18 | 8.0 | 1.7 |
| 2-D wave, onesided | 6.4e-9 | 0.26 | 14 | 14 | 11 | 4.5 |
| 2-D wave, cascade | 6.3e-7 | 8.0 | 24 | 20 | 13 | 1.7 |

**The root-edge undershoot.** With the filter on, the three-level nest under
global stepping is the one refined layout that undershoots, and the
undershoot starts at the first level's boundary plane. The case is the
two-species layer of `bench/interfaceconservation.jl` on that section's grid
with only the mass-fraction bound on, permissive, over two transits, sampled
eight times. The table gives the minimum mass fraction by distance from the
level-1 box edge, in root nodes on the root and in fine nodes from the
imposed plane on each fine level. The toggle changes no digit in any row,
since the bound is the only species sensor and it is zero inside [0, 1].

| nest, stepping | t = 37.7, root face node | root outside, 2–3 nodes | level-1 plane | t = 50.3, root face node | root outside, 4+ nodes | level-1 interior | level-2 interior |
|---|---|---|---|---|---|---|---|
| two levels, global | +1.6e-4 | +1.3e-4 | +1.6e-4 | +1.3e-4 | +3.9e-5 | +2.5e-4 | — |
| two levels, subcycled | +2.9e-4 | +1.3e-4 | +2.9e-4 | +2.6e-4 | +1.0e-4 | +4.1e-4 | — |
| two levels, global, no filter | −2.4e-4 | −2.9e-4 | −2.4e-4 | −4.7e-4 | −4.9e-4 | −3.6e-4 | — |
| three levels, global | −1.2e-4 | −2.5e-4 | −1.2e-4 | −5.1e-4 | −5.8e-4 | −3.4e-4 | −9.8e-5 |
| three levels, subcycled | +2.9e-4 | +1.3e-4 | +2.9e-4 | +2.6e-4 | +1.0e-4 | +4.1e-4 | +6.4e-4 |
| three levels, global, unrelaxed filter | −1.1e-4 | −2.5e-4 | −1.1e-4 | −5.0e-4 | −5.8e-4 | −3.4e-4 | −9.8e-5 |

A positive entry is a minimum above zero, so the two-level nests and both
subcycled nests carry no undershoot at any sample. Without the filter the
sheet undershoots on its own, at the box face and four or more root nodes
outside it alike, which is the unfiltered ringing the conservation
instrument's header describes and not an interface effect. In the
three-level nest under global stepping the undershoot appears first on the
coarse node coincident with the first level's boundary plane and the two or
three root nodes outside it, at t = 37.7, while the root interior and the
second level are still positive; by t = 50.3 the flow has carried it over
the whole root outside the box and through the first level. The unrelaxed
filter reproduces every entry to two digits. Direct weight measurements
below show that global stepping uses the finest directional rate on every
patch; the root pass is not relaxed in proportion to its own coarse spacing.

**Global-step follow-up (N12).** The global-step undershoot follows the
filter's application schedule. A second transfer is not necessary for its
onset, the frequency of fine-to-coarse restriction does not set it, and the
smaller timestep alone does not cause it. Reducing the two-level CFL from
0.45 to 0.15 matches the three-level root-step scale without adding another
interface, and the two-level run then develops an undershoot too. A
diagnostic driver that restricts fine data onto the parent every third step
retains the same RK stages, shell impositions and filter passes, and
produces almost the same undershoot. Keeping the small timestep but
filtering every third step restores the ordinary two-level physical filter
cadence and remains positive at every sample.

| two-level run, CFL 0.15 | t | completed steps | root face / level-1 plane | root outside, 4+ nodes | level-1 interior, 5+ nodes |
|---|---|---|---|---|---|
| production | 37.6991 | 7929 | +2.348e-4 | +8.906e-5 | +7.005e-4 |
| production | 43.9823 | 9251 | +3.027e-5 | +6.388e-5 | +7.047e-5 |
| production | 50.2655 | 10572 | -7.997e-5 | -2.395e-4 | +2.588e-4 |
| restriction every third step | 50.2655 | 10572 | -8.091e-5 | -2.397e-4 | +2.567e-4 |
| filter every third step | 50.2655 | | +1.347e-4 / +1.002e-4 | +3.852e-5 | +2.511e-4 |

The startup weights are identical on every patch of a hierarchy:

| levels | CFL | filter interval | dt | x weight | y weight |
|---|---|---|---|---|---|
| 2 | 0.45 | 1 | 0.0143184855 | 1 | 0.830373924 |
| 3 | 0.45 | 1 | 0.00478678860 | 1 | 0.827484401 |
| 2 | 0.15 | 1 | 0.00477282850 | 0.482471992 | 0.276791308 |
| 2 | 0.15 | 3 | 0.00477282850 | 1 | 0.830373924 |

Under global stepping the finest directional rate compensates for the
smaller timestep in the per-pass weight, while another level adds about
three times as many coarse-grid filter passes per unit physical time.
Reducing CFL on a fixed hierarchy also changes the number and weights of
the interleaved filter and shell operations; the x weight's cap makes even
the integrated relaxation strength different in these rows. The experiments
identify sensitivity to this filter schedule, not a CFL violation or a
restriction-frequency failure, and do not isolate an individual operator
commutator. Keep subcycling as the demonstrated remedy; the subsequent
[level-aware filter qualification](#bencharchivelevelfilterjl-level-aware-filtering-under-global-stepping)
measures candidate global-step policies and their accuracy tradeoffs.

Reproduce with `julia --project=. -t 1 bench/archive/interfacesensor.jl undershoot
layerN=96 layerny=24 layer_tfinal=50.26548245743669 samples=8 nmax=40000
undershoot_depths=2 "undershoot_variants=depth-3 dt" undershoot_ghosts=on`;
select `"undershoot_variants=sparse projection"` for the restriction
ablation or `"undershoot_variants==two levels, matched dt, filter every 3"`
for the filter cadence. The leading equals sign selects the exact label
instead of comma-separated substrings. These are controlled changes to
operation cadence, not a proposal to weaken synchronization in production.

**Decision.** The detector reads the ghost layers at an interface face
for every field recovered over the padded extent, the default; the strain
and dilatation sensors, computed on the interior, keep the clamp, and the
smoother keeps its closed-edge rows since its input has no ghosts. The
filter keeps the extended interface rows; the cascade rows are the worst
set at either kind of face. `detector = :d8` stays rejected in patched and
refined runs, since no interface rows exist for its pentadiagonal
operator. The closure candidates all pass the crossing gate, which
therefore promotes none of them. The step-on-a-plane failure under the
closure rows and the three-level undershoot under global stepping are
recorded as limitations: the first is an initial-data restriction with a
stated workaround, the second is removed by subcycling.

### bench/boundaryorder.jl idiv: the interface divergence rows

```text
julia --project=. -t 1 bench/boundaryorder.jl study=idiv
julia --project=. -t 1 bench/archive/interfacesensor.jl crossing "crossing_variants=base,idiv,reversed,two patches,three patches"
julia --project=. -t 4 bench/interfaceconservation.jl N=96 ny=24 tfinal=8.0 moving_tfinal=8.0 idiv=brady_livescu
```

`interface_divergence` replaces the flux divergence's closure rows at patch and level
interface ends and nowhere else; the derivative operator stays `lele_d1_6()`, so a wall keeps
the `:neutral3` rows. Without it an interface takes the `:cascade3` rows, and those rows below
reproduce [the matrix](walls.md#patch-interfaces-and-refinement-levels) to every digit.

On most smooth rows the Brady–Livescu rows raise the order at an interface from about three
or four under the default rows to five or six. The exceptions are the rows with walls, where the
`:neutral3` wall's fourth-order error dominates the window, and the one-sided viscous
gradients, which hold every source to about third order or less. `:cascade4` falls to second
order on the standing wave between two patches and below it between walls and an interface.
The `dt` column is below 0.1 on every row cited as an order unless marked.

```text
serial, Float64, cfl = 0.25, the interface window of the matrix
root N = 48, 96, 192 (49, 97, 193 for the wall rows)
```

```
                                            default                cascade4               Brady–Livescu
                                            N=192     orders       N=192     orders       N=192     orders
RHS on exact data, two patches, k=3         2.395e-5  3.36 / 3.25  3.546e-6  3.98 / 4.03  3.861e-7  5.27 / 5.41
RHS on exact data, 2 levels, k=3            1.849e-6  3.03 / 3.02  1.815e-8  4.23 / 4.14  2.435e-9  5.02 / 5.03
two patches, entropy k=3                    6.315e-7  5.10 / 4.71  2.716e-7  5.23 / 5.02  1.119e-8  6.63 / 6.96
two patches, entropy k=1                    8.262e-9  3.09 / 3.53  1.078e-9  5.64 / 4.97  8.611e-12 6.49 / 6.52
two patches, standing wave                  1.949e-8  3.41 / 3.87  6.000e-9  5.45 / 2.39  4.504e-11 5.77 / 5.82
two patches, standing wave, filtered        1.162e-8  3.61 / 3.85  1.183e-8  5.42 / time  5.129e-11 5.49 / 5.52
2 levels, entropy k=3                       9.408e-8  3.41 / 3.83  4.428e-10 5.71 / 3.92  6.401e-11 6.02 / 6.01
2 levels subcycled, entropy k=3             9.198e-8  3.43 / 3.84  4.475e-10 5.74 / 3.88  6.480e-11 6.02 / 5.99
3 levels, entropy k=3                       7.407e-8  3.50 / 3.89  4.481e-10 5.72 / 3.89  6.391e-11 6.02 / 6.01
3 levels subcycled, entropy k=3             7.192e-8  3.52 / 3.92  4.410e-10 5.75 / 3.87  6.491e-11 6.02 / 5.99
2 levels, tile 4, subcycled, entropy k=3    7.205e-8  3.99 / 3.72  1.572e-8  4.20 / 3.36  6.481e-11 6.02 / 5.99
2 levels, entropy k=3, filtered             1.577e-8  4.29 / 3.95  1.049e-10 7.01 / 6.80  7.378e-10 5.25 / 5.81
2 levels, entropy k=1                       7.380e-10 3.37 / 3.76  8.036e-12 4.31 / 4.93  5.174e-14 5.33 / time
2 levels, standing wave                     1.030e-9  3.37 / 3.47  9.571e-12 4.76 / 4.59  4.998e-13 5.13 / 5.11
walls and an interface, two patches         4.234e-10 5.04 / 5.66  1.322e-7  2.38 / 1.20  2.375e-9  4.56 / 5.19
walls and an interface, filtered            1.804e-10 4.41 / 4.88  1.056e-7  2.52 / −2.88 1.217e-10 5.21 / 4.00
viscous, two patches, extended gradients    2.469e-8  3.62 / 3.87  1.419e-9  4.87 / 4.99  6.850e-11 4.92 / 5.66
viscous, two patches, one-sided gradients   2.470e-6  1.91 / 0.91  4.788e-6  0.49 / 0.23  4.770e-6  0.84 / 0.16
viscous, 2 levels, extended gradients       8.597e-10 3.61 / 3.65  7.309e-12 4.85 / 4.75  3.511e-13 5.57 / 5.34
viscous, 2 levels, one-sided gradients      3.455e-9  3.44 / 3.66  3.652e-10 2.92 / 3.07  4.927e-10 3.25 / 3.30
```

At half the step the Brady–Livescu standing wave through two levels stays at fifth order: it
converges at 5.08 and then reaches the round-off floor.

| two levels, standing wave, `cfl = 0.125` | N = 96 | N = 192 | N = 384 |
|---|---|---|---|
| Brady–Livescu | 2.022e-11 | 3.459e-13 | 6.084e-14 |
| Brady–Livescu, `dt` column | | 0.004 | |
| default | 1.109e-8 | 7.778e-10 | 7.504e-11 |

On the walls-and-interface rows the wall window carries the `:neutral3` wall's fourth-order
error, 3.1e-9 under the default rows and 2.8e-9 under Brady–Livescu on the finest grid, and
the interface window reads that error.

In single precision no row converges and no source lowers the floor. The error of the
single-wavelength entropy wave in the interface window is 3e-6 to 6e-5 at every N and under
every source, rising with N.

The Brady–Livescu rows reflect least from a same-level face on the two finer grids. On the
nests the three sources reflect alike on the coarsest grid, and on the finest the default
reflects about ten times more than the other two. The entry is the acoustic pulse of the
reflection tests, the left-running characteristic upstream of the first face over the pulse
amplitude, against the run without the face:

```
                      N = 96                           N = 192                          N = 384
                      default   cascade4  BL           default   cascade4  BL           default   cascade4  BL
two patches           9.740e-6  6.060e-5  1.543e-5     2.049e-6  8.356e-6  1.139e-6     2.685e-7  1.897e-6  2.477e-8
2 levels              1.182e-6  1.241e-6  1.189e-6     1.188e-8  3.062e-9  1.240e-9     8.427e-10 1.046e-10 6.066e-11
2 levels subcycled    8.314e-7  8.928e-7  8.933e-7     1.051e-8  2.897e-9  1.099e-9     7.058e-10 8.713e-11 5.242e-11
```

**Shocks.** Neither candidate survives a discontinuity on a shared plane, and the
Brady–Livescu rows halve the minimum pressure where the shock crosses a same-level face. The
case is the Sod crossing of
[the interface sensor instrument](#bencharchiveinterfacesensorjl-the-sensors-and-the-filter-at-an-interface),
root N = 201, `cfl = 0.4`, the artificial properties and the filter live; shell and root
columns are the density error against the uniform run at t = 0.2:

| row | default | `:cascade4` | Brady–Livescu |
|---|---|---|---|
| two levels subcycled: steps, p_min, shell, root | 373, 0.0405, 7.7e-3, 2.4e-2 | 373, 0.0405, 7.9e-3, 2.4e-2 | 373, 0.0405, 8.2e-3, 2.6e-2 |
| global dt: p_min, shell | — | 0.0382, 9.0e-3 | 0.0382, 1.0e-2 |
| three levels: shell | — | 6.5e-3 | 7.0e-3 |
| tile 8: p_min, shell | — | 0.0405, 8.3e-2 | 0.0206, 7.8e-2 |
| three patches: p_min, shell, root | 0.0412, 7.4e-2, 2.7e-2 | 0.0412, 7.7e-2, 2.7e-2 | 0.0205, 6.8e-2, 5.5e-2 |
| diaphragm on the shared plane | negative density, step 2 | step 1 | step 1 |
| the same, one-sided gradients | completes, p_min 0.097 | negative density, step 113 | step 2 |

The mirrored tube, the shock running right to left through the same faces, reproduces every
printed digit of the forward rows. Under either candidate the one-sided gradient rows no
longer serve as a workaround for a discontinuity on a shared plane.

**Conservation.** The nests and the regrid histories agree across the three sources to the
second digit, all within the budget. At a same-level face the Brady–Livescu rows drift least
and leave no mass-fraction excursion, and `:cascade4` drifts most. The case is the layer of
[the conservation instrument](#benchinterfaceconservationjl-composite-conservation-budgets)
on that section's grid to t = 8.

| layer to t = 8, every source | |
|---|---|
| fixed-nest drift | 6.9e-5 to 7.1e-5 |
| cumulative regrid jump | 2.2e-4 |
| budget | 1e-3 |

| same-level layout, layer to t = 8 | default | `:cascade4` | Brady–Livescu |
|---|---|---|---|
| drift | 2.9e-10 | 3.3e-9 | 1.5e-11 |
| largest mass-fraction excursion | 1.6e-4 | 1.6e-3 | 0 |

**Decision.** The default stays. `:cascade4` is rejected: the undamped mode it carries at an
inviscid wall appears at an interface on the standing wave and grows between walls and an
interface. The Brady–Livescu rows reach about sixth order on the entropy waves at every
interface, but 5.1 on the acoustic standing wave at a coarse-fine face, their formal fifth
order, below the 5.5 target. They halve the minimum pressure of a shock crossing a
same-level face. The option
stays experimental, a Float64 smooth-flow setting.

### bench/boundaryorder.jl gflux: the divergence through interface ends from ghost fluxes

```text
julia --project=. -t 1 bench/boundaryorder.jl study=gflux
julia --project=. -t 1 bench/archive/interfacesensor.jl crossing
julia --project=. -t 1 bench/archive/interfacesensor.jl startup
julia --project=. -t 4 bench/interfaceconservation.jl N=96 ny=24 tfinal=8.0 moving_tfinal=8.0 iflux=ghost [mu=1e-3]
julia --project=. -t 1 bench/archive/interfacecost.jl
```

`interface_flux = :ghost` evaluates the inviscid flux on the padded block, the interface ghosts
included, and differences it through the gradient plans' interface rows. With the artificial
properties off, as in every smooth row of this section, the molecular flux joins it: its ghost
values at a same-level face are the neighbouring patch's own interior flux, exchanged after
every patch of the level has evaluated, and at a coarse-fine face they are evaluated from the
conserved gradients of the interpolated fine box, taken with explicit seventh-order rows at
the box ends. The wall corrections then take the one-sided rows of `div_plans`, the cascade
rows or an `interface_divergence` source's. With the artificial properties on, the remainder
F − f (the molecular and artificial fluxes and the wall corrections) is carried in
`ghost_flux` as well, exchanged at a same-level face and extrapolated at a coarse-fine face,
and the whole flux takes the gradient rows (the remainder paragraph below). The ghost column
runs at the ghost path's default level interpolation order, 8.

The ghost fluxes take the interface out of the error budget. At a same-level face the
interface window's error equals the interior's, and at a coarse-fine face every row matches
or leads the Brady–Livescu (BL) rows. The polynomial row, `polynomial_case(96)` with every
inviscid flux component of degree at most five, is differenced to round-off in the interface
window by the ghost path alone. The closure-row and BL columns are those of the section
above, which this study reproduces to every digit. Package at `b4388f9`, Julia 1.13.3; the
transcript of the first command is `bench/archive/results/boundaryorder.txt`.

```text
serial, Float64, cfl = 0.25, the matrix's interface window
root N = 48, 96, 192 (49, 97, 193 for the wall rows)
```

```
                                            closure                BL                     ghost
                                            N=192     orders       N=192     orders       N=192     orders
polynomial RHS, two patches (ρu, E)         2.09e-8, 1.06e-8       4.58e-12, 9.25e-13     2.99e-14, 2.18e-14
polynomial RHS, 2 levels (ρu, E)            7.73e-10, 4.20e-10     5.68e-12, 1.01e-11     1.58e-13, 1.81e-13
RHS on exact data, two patches, k=3         2.395e-5  3.36 / 3.25  3.861e-7  5.27 / 5.41  1.785e-9  5.97 / 5.99
RHS on exact data, 2 levels, k=3            1.849e-6  3.03 / 3.02  2.435e-9  5.02 / 5.03  1.280e-10 6.39 / 6.00
two patches, entropy k=3                    6.315e-7  5.10 / 4.71  1.119e-8  6.63 / 6.96  1.416e-10 6.75 / 6.78
two patches, entropy k=1                    8.262e-9  3.09 / 3.53  8.611e-12 6.49 / 6.52  6.950e-14 6.88 / 6.70
two patches, standing wave                  1.949e-8  3.41 / 3.87  4.504e-11 5.77 / 5.82  4.308e-12 5.44 / time
two patches, standing wave, filtered        1.162e-8  3.61 / 3.85  5.129e-11 5.49 / 5.52  4.440e-12 5.34 / time
2 levels, entropy k=3                       9.408e-8  3.41 / 3.83  6.401e-11 6.02 / 6.01  6.380e-11 6.02 / 6.01
2 levels subcycled, entropy k=3             9.198e-8  3.43 / 3.84  6.480e-11 6.02 / 5.99  6.481e-11 6.02 / 5.99
3 levels, entropy k=3                       7.407e-8  3.50 / 3.89  6.391e-11 6.02 / 6.01  6.379e-11 6.02 / 6.01
3 levels subcycled, entropy k=3             7.192e-8  3.52 / 3.92  6.491e-11 6.02 / 5.99  6.481e-11 6.02 / 5.99
2 levels, tile 4, subcycled, entropy k=3    7.205e-8  3.99 / 3.72  6.481e-11 6.02 / 5.99  6.481e-11 6.02 / 5.99
2 levels, entropy k=3, filtered             1.577e-8  4.29 / 3.95  7.378e-10 5.25 / 5.81  1.049e-10 6.96 / 6.78
2 levels, entropy k=1                       7.380e-10 3.37 / 3.76  5.174e-14 5.33 / time  3.431e-14 6.01 / 5.77
2 levels, standing wave                     1.030e-9  3.37 / 3.47  4.998e-13 5.13 / 5.11  1.666e-13 6.12 / 5.94
2 levels subcycled, standing wave           9.640e-10 3.37 / 3.47  1.519e-11 4.70 / time  1.938e-11 4.84 / time
walls and an interface, two patches         4.234e-10 5.04 / 5.66  2.375e-9  4.56 / 5.19  1.525e-10 4.27 / 4.98
walls and an interface, filtered            1.804e-10 4.41 / 4.88  1.217e-10 5.21 / 4.00  1.317e-10 4.90 / 4.12
viscous, two patches                        2.469e-8  3.62 / 3.87  6.850e-11 4.92 / 5.66  3.164e-13 6.75 / 6.43
viscous, 2 levels                           8.597e-10 3.61 / 3.65  3.511e-13 5.57 / 5.34  1.070e-13 6.29 / 6.04
viscous, 2 levels subcycled                 8.529e-10 3.61 / 3.64  3.400e-13 5.53 / 5.39  1.057e-13 6.28 / 6.06
```

The equality of the interface window and the interior at a same-level face holds on every
ghost row, viscous ones included. The inviscid standing-wave slopes there are set by the time
integrator, the `dt` column reading 0.83–0.86 on the finest grid. The BL rows on the
remainder change nothing without the artificial properties: the ghost path leaves no
remainder, and the `ghost+BL` column repeats the ghost column to every digit.

| two patches, ghost | at the face | inside |
|---|---|---|
| entropy wave k = 3 | 1.42e-10 | 1.55e-10 |

At a coarse-fine face the interpolation order sets the ghost path's accuracy. The ghost
state is the interpolation of the parent, whose O(h^p) value error enters the differenced
inviscid flux at h^(p−1) and the molecular flux at h^(p−2). At order 6, below the default,
the right-hand side on exact data converged at 5.01, the BL rows' order, with a constant
eight times larger, and the evolution errors sat five to twelve times above the BL rows.

Near an interface end the viscous right-hand side converges at about sixth order against the
periodic operator at the patch's spacing, through two patches and through a level. The
measure is the error within four nodes of an interface end on exact data, the check of
`test/patch_tests.jl` on one species.

| viscous right-hand side on exact data, N = 48 to 384 | orders |
|---|---|
| two patches | 5.92 / 5.98 |
| a level | 6.73 / 5.44 / 7.02 |
| a level, order 10 | 6.23, then the 1e-12 floor |

Under the third-order closure rows of the operator's own set at the box ends, the level row
stalls at a second-order floor of 5e-11. The box ends lie eight to eleven fine nodes from the
ghost layers, and a closure row's error decays by 2 − √3 per node.

In single precision the ghost path keeps the floor of the other sources on the
single-wavelength entropy wave: 4.4e-6 through two patches and 8.4e-6 through two levels on
the finest grid.

The ghost fluxes reflect least from the interface on the two finer grids in every layout but
the subcycled level at N = 192, and through two patches at N = 384 forty times less than the
BL rows. The entry is the acoustic pulse of the reflection tests, the left-running
characteristic over the amplitude:

```
                      N = 96                           N = 192                          N = 384
                      closure   BL        ghost        closure   BL        ghost        closure   BL        ghost
two patches           9.740e-6  1.543e-5  1.540e-6     2.049e-6  1.139e-6  2.049e-8     2.685e-7  2.477e-8  5.775e-10
2 levels              1.182e-6  1.189e-6  9.773e-7     1.188e-8  1.240e-9  9.675e-10    8.427e-10 6.066e-11 2.031e-11
2 levels subcycled    8.314e-7  8.933e-7  4.824e-7     1.051e-8  1.099e-9  1.417e-9     7.070e-10 5.213e-11 5.072e-11
```

**Shocks.** The ghost fluxes keep the closure rows' minimum pressure on every layout measured
under both, and they complete the run with the diaphragm on a shared plane, which fails under
the closure rows. The case is the Sod crossing, root N = 201, `cfl = 0.4`, the artificial
properties and the filter live, inviscid, at the ghost path's default order 8. Package at
`b4388f9`, Julia 1.13.3; the transcript of the `crossing` and `startup` commands is
`bench/archive/results/interfacesensor.txt`. Shell and root are the density errors against the
uniform run at the fine spacing at t = 0.2, p_min the smallest pressure on any step. In the
default ghost column the remainder goes through the gradient rows (the transcript's
`rem-extrap` rows on a nest, `grad-rem` on patches); in the middle column it takes the
one-sided rows, both toggles off (the transcript's `gflux` rows):

| row | closure | ghost, one-sided remainder | ghost (default) |
|---|---|---|---|
| two levels subcycled: steps, p_min, shell, root | 326, 0.0405, 9.8e-3, 2.8e-2 | 326, 0.0405, 9.5e-3, 2.8e-2 | 326, 0.0405, 8.4e-3, 2.8e-2 |
| global dt: steps, p_min, shell | — | 708, 0.0382, 9.1e-3 | 707, 0.0382, 8.7e-3 |
| three levels: p_min, shell | — | 0.0348, 8.9e-3 | 0.0348, 7.9e-3 |
| three levels global dt: p_min, shell | 0.0253, 7.4e-3 | 0.0253, 7.9e-3 | 0.0253, 7.7e-3 |
| tile 8: p_min, shell | — | 0.0405, 7.6e-2 | 0.0387, 7.6e-2 |
| three patches: p_min, shell, root | 0.0412, 7.5e-2, 2.7e-2 | 0.0412, 7.9e-2, 2.9e-2 | 0.0412, 7.9e-2, 2.7e-2 |
| diaphragm on the shared plane | negative density, step 2 | completes: 304 steps, p_min 0.075, shell 1.1e-2, root 9.6e-2 | completes: 306 steps, p_min 0.088, shell 1.0e-2, root 8.2e-2 |
| the same, one-sided gradient workaround | completes: p_min 0.097, shell 4.6e-2, root 2.35e-1 | | |
| the same, art off (the toggles inert) | — | completes: 236 steps, p_min 0.053, shell 2.9e-2, root 8.9e-2 | the same |
| two species: shell Y excursion, elsewhere | — | 1.1e-9, 1.6e-4 | 1.3e-9, 1.6e-4 |

The reversed tube reproduces every printed digit of the forward ghost rows under either
remainder.

A Sod diaphragm on the plane between two tiles exercises the positivity failsafe, which
repairs half again as many cells under the ghost fluxes as under the closure rows. The
flagged cells are the start-up undershoot of an ideal jump, which the scheme produces
without an interface; the interface paths differ in how far they damp that transient. The
failsafe's case in `test/level_tests.jl` places the diaphragm on that plane, and the
columns are the `low_energy`, `steps` and `cells` fields of the solver's `floor_tally`:

```text
root N = 81, level-1 tiles of 8 over root nodes 24–56, cfl = 0.2, 20 steps,
floor_ratio = 0.999
```

| failsafe at `743fdb4b` | cells below the internal-energy floor | steps on which it acted | repaired |
|---|---|---|---|
| ghost fluxes (default) | 116 | 20 | 113 |
| ghost, remainder on the one-sided rows | 133 | | 141 |
| ghost, coarse-fine ends not extrapolated | 116 | | 113 |
| closure rows | 63 | 18 | 70 |

At `aa5757b`, with the remainder on the one-sided rows, the repaired cells lay 1–11 fine
nodes into the low-density tile and density triggered four in five of them. The
interpolation order, 6 or 8, left the tallies unchanged on both paths. The `startup` part
counts the same criterion without repairing (cells below 0.999 of the initial minimum
density or total energy over density, summed over steps) and gives the level-1 density error
against the uniform run at the fine spacing. In the table, `ghost` is the ghost path with
both remainder toggles off and the last two paths run with them on, the second of the two
being the default:

| diaphragm | path | 20 steps: flagged, min ρ/ρ0, L1(ρ) | t = 0.12: flagged, min ρ/ρ0, L1(ρ) |
|---|---|---|---|
| none (uniform, fine spacing) | — | 173, 0.671, — | 890, 0.671, — |
| on the tile plane | ghost | 150, 0.745, 3.31e-3 | 1965, 0.745, 2.35e-3 |
| | closure | 114, 0.970, 1.35e-3 | 1708, 0.762, 2.81e-3 |
| | ghost, BL remainder rows | negative density, step 3 | — |
| | ghost, gradient-row remainder | 165, 0.988, 2.91e-3 | 1926, 0.760, 3.00e-3 |
| | the same, coarse-fine ends extrapolated | 165, 0.988, 2.91e-3 | 1922, 0.760, 3.02e-3 |
| mid-tile | ghost | 145, 0.671, 6.36e-5 | 2246, 0.671, 6.03e-4 |
| | closure | 156, 0.671, 9.58e-5 | 2300, 0.671, 1.44e-3 |
| | ghost, gradient-row remainder | 145, 0.671, 6.36e-5 | 2210, 0.671, 6.13e-4 |
| | the same, coarse-fine ends extrapolated | 145, 0.671, 6.36e-5 | 2226, 0.671, 5.73e-4 |
| on a coarse-fine face | ghost | 176, 0.556, 9.93e-3 | 2903, 0.556, 1.39e-2 |
| | closure | 106, 0.556, 8.20e-3 | 2804, 0.556, 1.46e-2 |
| | ghost, remainder, coarse-fine ends extrapolated | 173, 0.556, 9.67e-3 | 2893, 0.556, 1.40e-2 |

On a tile plane the default ghost path and the closure rows damp the start-up transient to
about the same minimum density, and the ghost path with the remainder on the one-sided rows
damps it least. On the uniform grid the start-up minimum is reached at step 8. A jump on a
tile plane under the closure rows starts through the one-sided inviscid and remainder rows,
whose dissipation damps that transient. The ghost fluxes with the remainder on the
one-sided rows difference the inviscid flux there with interior accuracy, and keep their
start-up minimum to t = 0.12. With the remainder through the gradient rows as well, the
default, the transient is damped at least as much as under the closure rows, and at
t = 0.12 the minimum of both is a later one on the root. A diaphragm inside a tile or on a
coarse-fine face gives the same minima under every path.

The Brady–Livescu rows on the remainder have failed since `1fc1fdb`, which computes a tiled
level's artificial coefficients over the whole level. At `c78eee8` they completed with a
lower minimum and a larger error than the ghost path:

| on the tile plane, at `c78eee8` | min ρ/ρ0, L1(ρ) |
|---|---|
| ghost, BL remainder rows | 0.664, 7.39e-3 |
| ghost | 0.770, 2.89e-3 |

On a root grid of twice the resolution the flagged counts rise less than twofold and the
minima are unchanged. On the three regridded levels of `test/level_tests.jl` (shock and
contact, subcycled, run in the test's four calls) the two paths flag nearly the same count.

| at `aa5757b`, t = 0.0045 | flagged at root N = 161 over N = 81 |
|---|---|
| ghost | 1.8-fold |
| closure | 1.4-fold |

| three regridded levels to t = 0.25, at `743fdb4b` | flagged | min ρ/ρ0 |
|---|---|---|
| ghost (default) | 3750 | 0.663 |
| closure | 3744 | 0.675 |

**The remainder through the gradient rows.** Differenced through the gradient rows, the
remainder raises the interface's order with the artificial properties on from about three
to six. In the plain form the remainder F − f of the ghost-differenced flux (the
molecular and artificial fluxes and the wall rewrite) takes the divergence plans' one-sided
rows at an interface end, a second line solve. The second column's form carries it in `ghost_flux`
on dimensions whose interface ends are same-level, the level's records filling the ghost
layers, and differences the whole flux through the gradient plans in one solve; the third
column's extends that to coarse-fine ends, whose ghost layers take a degree-5 Lagrange
extrapolation of the interior remainder. The solver takes the third form, the
extrapolation applied per tile, so a stacked device level takes it too. The table is on exact
data with prescribed smooth μ\*, β\*, κ\* (`compute_rhs!` with `coefficients_current`), in
the interface window on the section's three grids:

| layout | plain | same-level ends | and coarse-fine ends |
|---|---|---|---|
| two patches, inviscid | 6.95e-5, 6.33e-6, 6.90e-7 (3.46, 3.20) | 1.75e-6, 2.91e-8, 4.62e-10 (5.91, 5.98) | the same |
| two patches, μ = 2e-2 | (3.45, 3.20) | (5.91, 5.98) | the same |
| two levels, inviscid | (2.76, 2.97) | the same as plain | (5.99, 5.93), 1.28e-10 at N = 192 |
| two levels, μ = 2e-2 | (2.79, 2.97) | the same as plain | (5.96, 5.99) |

On shocks the ghost-differenced remainder lowers the error at the shell slightly and leave the rest of the
solution as it was. The comparison is the third column's form against the plain one on the `crossing` rows of
the transcript above, np = 1, at t = 0.2. On the untiled nests the shell error falls by 2 to
12%. The fine and root errors there move by under 0.3%, and no row has an inadmissible step.

| `crossing` row | quantity | plain | ghost-differenced |
|---|---|---|---|
| base | shell | 9.50e-3 | 8.42e-3 |
| base | fine | 7.571e-2 | 7.571e-2 |
| base | root | 2.768e-2 | 2.761e-2 |
| three levels | shell | 8.90e-3 | 7.88e-3 |
| global step | shell | 9.07e-3 | 8.66e-3 |
| tile 8 | level-1 error (`fine` column) | 9.76e-3 | 9.04e-3 |
| two patches | shell | 1.05e-2 | 1.01e-2 |
| two patches | root | 9.55e-2 | 8.19e-2 |
| three patches | shell | 7.95e-2 | 7.86e-2 |

On `test/validation.jl`'s Noh with a level at the axis and the coarse-fine correction on, the
toggles leave the run holding the shock unchanged in plateau and mass drift. The run passing
the shock out moves slightly, its most negative internal energy rising.
`bench/archive/couplingspectrum.jl` (art off) and `bench/interfaceconservation.jl iflux=ghost`'s
mixing layer, whose remainder is zero, are unchanged to every printed digit.

| Noh at the axis, at `d346d35` | plateau | mass drift | steps | tile error | shock radius | e_min |
|---|---|---|---|---|---|---|
| holding the shock, neither toggle | 15.6523 | −9.7e-7 | 6641 | 5.693e-3 | | |
| holding the shock, both toggles | 15.6523 | −9.7e-7 | 6639 | 5.658e-3 | | |
| passing the shock out, neither toggle | 14.9341 | −4.7e-7 | | | 0.2092 | −0.0189 |
| passing the shock out, both toggles | 14.9059 | −3.6e-7 | | | 0.2094 | −0.0162 |

**Conservation.** At a same-level face the ghost fluxes conserve to round-off with no
mass-fraction excursion, and every nest and regrid history agrees with the closure rows to
the third digit. The case is the layer on that section's grid to t = 8, inviscid and with
`mu=1e-3` on every layout, the uniform run included.

| layer to t = 8 | ghost, inviscid | closure, inviscid | ghost, `mu=1e-3` | closure, `mu=1e-3` |
|---|---|---|---|---|
| same-level drift | 3.9e-13 | 2.9e-10 | 3.9e-13 | 2.8e-10 |
| same-level mass-fraction excursion | 0 | 1.6e-4 | 0 | 6.0e-5 |
| nest drift | | | 6.75e-5 to 6.92e-5 | 6.76e-5 to 6.93e-5 |
| regrid histories, summed | | | 2.24e-4 | 2.24e-4 |

**Cost.** With the artificial properties on, the ghost fluxes cost between a sixth and a half
more per step than the closure rows, the most on a subcycled tiled level. With them off, the
gradient ring built at every shell imposition doubles the cost on that level.
`bench/archive/interfacecost.jl`: serial, one thread, Float64, the median step over nine rounds of a
few steps per configuration, the ghost and closure solvers of a round stepping in alternating
order in one process, periodic, with the artificial properties and the filter on, at the
default level interpolation order of each path. Package at the commit that builds the shell's
gradient ring only where a coarse-fine face reads it, Julia 1.13.3. The machine is the Core
i9-12900K, and the transcript is `bench/archive/results/interfacecost.txt`.

| ghost over closure, per step | ratio |
|---|---|
| (64, 48, 48) in two patches, inviscid | 1.170 |
| (64, 48, 48) in two patches, `mu = 1e-3` | 1.250 |
| 2-D static 17² box on N = 96, global step, `mu = 1e-3` | 1.165 |
| 2-D subcycled level of sixteen 37-node tiles on N = 96, `mu = 1e-3` | 1.525 |
| 3-D static 13³ box on N = 40, global step, `mu = 1e-3` | 1.351 |
| the sixteen-tile level, artificial properties off | 2.104 |

With the artificial properties on, the extrapolated remainder carries the molecular flux at a
coarse-fine face and no gradient ring is built. With them off the ring is built and filled at
every shell imposition (the compact derivatives of every component along every dimension over
the lines of the interpolated box that cross the ring). One imposition of the sixteen-tile
level's shells then takes 1.447 ms against 0.329 ms without the ring. The `ghost_flux` arrays
add one state-sized array per interface dimension of a patch.

**Decision.** Same-level interfaces: the ghost fluxes remove the interface from the error
budget, inviscid and viscous (the interface window equals the interior at every
order-measuring row), cut the acoustic reflection forty-fold below the BL rows at N = 384, keep
the closure rows' shock minimum pressure, conserve to round-off without a mass-fraction excursion,
and are the first treatment under which the diaphragm on a shared plane runs with the extended
gradients. Coarse–fine faces: at the ghost path's default order it reads 5.8–7.0 on every
smooth row the time integrator leaves, viscous included, and matches or leads the BL rows
everywhere, so one treatment serves both interface kinds. It is the default (ROADMAP N15b).

### bench/archive/levelfilter.jl: level-aware filtering under global stepping

N12a compares the production filter with two benchmark-only policies. The
driver explicitly loads `bench/archive/level_filter_policy.jl` in a fresh process;
it does not install a solver option or change the package default. These
are Float64 CPU measurements on an Intel Core i9-12900K, one Julia thread
per process. The Julia version is 1.11.4. The helper adds a rate sweep and a
collective; no performance conclusion is drawn from it.

For patch level `l`, the normalized policy forms the globally reduced
directional envelope `q_d = max(r_patch,d / 3^l)` and uses `3^l q_d` in
that patch's existing relaxed filter weight. The CFL rate that sizes the
step is unchanged. The reduction includes all ranks, with zeros from
nonowners, and is rebuilt at every rate estimate, including the first
estimate after regridding. This is the same conservative envelope used by
subcycling, expressed in each patch's physical time units.

The cadence comparator uses that envelope but filters level `l` only
every `3^(L-1-l)` global steps, multiplying its weight by the same stride;
`L` is the number of levels. It is restricted to `filter_interval=1`.
It multiplies the latest timestep rather than accumulating elapsed time,
so shortened steps and an unfinished final stride are not a general
physical-cadence prescription. Both trials delegate subcycling and
`filter_cfl=0` to production unchanged. The helper also provides an
actual-per-level envelope for diagnostic checks; that variant is outside
the qualification matrix below.

**Two-transit layer and composite budgets.** Both candidates remove the
sampled late undershoot while remaining well inside the conservation
allowance. The conservation instrument runs the bound-only two-species
layer on `(96,24,1)` to `t=50.26548245743669`, with sixteen equally spaced
observations, CFL 0.45
and permissive validity. Each drift is relative to that layout's own
initialized composite quadrature. The evolution budget remains 1e-3 for
mass, each species mass, energy and momentum scaled by initial mass times
sound speed; it was not adjusted for these candidates.

| policy, global stepping | levels | steps | maximum sampled conserved drift | maximum species excursion |
|---|---|---|---|---|
| production | 2 | 3532 | 7.052e-7 | 0 |
| normalized | 2 | 3532 | 4.880e-7 | 0 |
| cadence | 2 | 3532 | 4.848e-7 | 0 |
| production | 3 | 10568 | 8.495e-7 | 3.64e-4 |
| normalized | 3 | 10568 | 4.773e-7 | 0 |
| cadence | 3 | 10568 | 4.803e-7 | 0 |

Subcycling remains positive at both depths, and its production maximum
drifts are 4.851e-7 and 4.847e-7. The policies were run at 8 ranks, 5.3
minutes each. The agreement of the two candidate layer outcomes does not
make their operators equivalent: one applies a weak pass at every
fine-sized step, the other interleaves fewer stronger passes with shell
imposition.

Reproduce with `julia --project=. -t 1 bench/archive/levelfilter.jl
policy=normalized instrument=budgets N=96 ny=24
tfinal=50.26548245743669 moving_tfinal=50.26548245743669 samples=16
check=true`. Use `parts=mixing layouts=uniform,depth3 stepping=global`
to isolate the long three-level rank comparison while retaining its
uniform attribution baseline.

**Moving refinement.** Every conservation measure passes, but moving
refinement still has a species excursion beyond the validity dead band
under either trial, as under the unchanged subcycled control. These
policies therefore resolve the fixed-layer reproducer, not general species
positivity. The sharper moving layer uses the same grid and end time, with
sixteen explicitly sampled regrids of a two-level nest. The table separates
evolution from the immediate transfer jumps, using the existing 1e-3
allowance for each reported conserved-budget measure.

| policy, global stepping | sum of absolute transfer jumps | maximum total drift | maximum drift with transfers subtracted | species excursion |
|---|---|---|---|---|
| production | 3.824e-6 | 1.708e-5 | 1.467e-5 | 7.06e-4 |
| normalized | 5.948e-6 | 1.406e-5 | 1.027e-5 | 5.64e-4 |
| cadence | 5.710e-6 | 1.254e-5 | 8.908e-6 | 5.71e-4 |
| production, subcycled (control) | 5.801e-6 | 1.259e-5 | 8.927e-6 | 5.76e-4 |

Three-level moving refinement is unsupported and is not measured.

**Smooth evolution and reflection.** Both trials bring the global step
close to the subcycled order at both depths and lower its finest-grid L2
error, by about a third at two levels and nearly three quarters at three.
The shared entropy-wave case runs to `t=0.5` at root `N=48,96,192`, CFL
0.25 and 0.125, with the filter on. The interface maximum counts actual
patch faces, not MPI block ends; the L2 norm uses the solver's composite
quadrature, the coupling's conserved one. Serial, about 15 seconds per
policy. The following orders fit all three spacings at CFL 0.25.

| policy, global stepping | levels | interface order | L2 order | L2 error, N=192 |
|---|---|---|---|---|
| production | 2 | 7.293 | 6.860 | 6.829e-11 |
| normalized | 2 | 7.758 | 6.450 | 4.514e-11 |
| cadence | 2 | 7.755 | 6.444 | 4.512e-11 |
| production | 3 | 7.161 | 6.992 | 1.643e-10 |
| normalized | 3 | 7.751 | 6.451 | 4.513e-11 |
| cadence | 3 | 7.773 | 6.449 | 4.510e-11 |

The subcycled orders are about 7.91 at the interface and 6.44 in L2.
Under the global step, halving the CFL changes no error by more than a
fraction of the accuracy instrument's 10% temporal-contamination criterion.
Subcycled, the change passes it only on the finest grid, where the
interface error is 6e-12 and the step's own error shows. These measured
orders do not identify an individual operator mode.

| largest relative change on halving CFL | |
|---|---|
| production, global step | 0.18% |
| normalized, global step | 0.49% |
| cadence, global step | 1.6% |
| subcycled, up to N = 96 | 0.8% |
| subcycled, N = 192 | 21% |

Every policy reflects the filtered acoustic pulse far inside the
pre-existing 1% reflection bound, the two candidates about half as much as
production. The pulse crosses both faces of the nest at root `N=192`,
measured at `t=pi/sqrt(1.4)` against a filtered uniform run. The entry is
the serial maximum of the upstream pressure-wake and
leftgoing-characteristic errors, divided by pulse amplitude, over two and
three levels and both stepping modes.

| policy | reflection |
|---|---|
| production | 3.26e-9 |
| normalized | 1.53e-9 |
| cadence | 1.46e-9 |

Subcycled accuracy and reflection rows are unchanged by the trial hooks.
Reproduce with `bench/archive/levelfilter.jl policy=normalized instrument=accuracy
parts=all ns=48,96,192` after `julia --project=. -t 1`.

**Shock crossings.** Under global stepping both candidates raise the
minimum density and pressure of a shock crossing and lower the disturbance
ahead of it, the most at three levels. The sensor instrument's Sod shock
enters and exits the refinement box by `t=0.2`, root `N=201`, with a
uniform fine-spacing reference. All listed runs completed with zero inadmissible steps. The
minimum density and pressure include startup; the disturbance is the
momentum ahead of the shock at `t=0.1`, `x>0.85`.

| policy, global stepping | levels | minimum density | minimum pressure | disturbance ahead of shock |
|---|---|---|---|---|
| production | 2 | 0.06239 | 0.03822 | 3.543e-10 |
| normalized | 2 | 0.07956 | 0.05828 | 1.216e-10 |
| cadence | 2 | 0.07670 | 0.05181 | 2.210e-10 |
| production | 3 | 0.05121 | 0.02518 | 4.446e-9 |
| normalized | 3 | 0.08174 | 0.06098 | 5.048e-11 |
| cadence | 3 | 0.07740 | 0.05315 | 2.403e-10 |

The two-species global-step crossing also stays admissible under every
policy, and both candidates lower its final mass-fraction excursion, the
most at the shell.

| two species, global step | excursion at the contact | excursion at the shell |
|---|---|---|
| production | 4.951e-4 | 6.953e-8 |
| normalized | 3.990e-4 | 1.328e-8 |
| cadence | 3.909e-4 | 6.283e-9 |

The subcycled two- and three-level rows reproduce production under both
trials. This tests passage into and out of a refined region; it does not
remove the separate discontinuity-on-a-same-level-plane limitation.

Reproduce the crossing rows with
`julia --project=. -t 1 bench/archive/levelfilter.jl policy=normalized
instrument=sensors parts=crossing N=201
"crossing_variants=base,global,three levels,two species"`, substituting
`policy=default` or `policy=cadence` for the controls.

**Rank coverage.** The candidates' results agree across every rank count
tried. The normalized three-level global-step layer was run over both
transits at one, two, four and eight ranks. Every run took 10568 steps,
with zero species excursion, matching printed width and molecular-mixing
histories, and the serial run's maximum sampled conserved drift. That drift
is 7.578e-5 in the quadrature of commit `fbb3ad3`, before the junction
correction Ω entered it; the budget table above is the later measurement. The
conservation, initial-sampling and mixing-comparison checks all pass.

All three policies' smooth and reflection matrices were repeated at four
and eight ranks, with the normalized matrix also repeated at two ranks.
Eight ranks use `ns=96,192`, since a 48-node root leaves a rank fewer
nodes than the filter's nine. Over common resolutions, the largest absolute
difference from serial in any reported smooth error norm is below
3.6e-14. Every reflection row passes. The standalone
`bench/archive/level_filter_policy_check.jl` checks default delegation, shortened
steps, skipped and due cadence passes, subcycle trajectory equality, and
propagation of a fastest rate present only on deep-level owners to root
ranks that own no deep tile.

The policy checks pass at one, two, four and eight ranks. Normalized-policy
moving regrid smoke runs pass at four and eight ranks in both stepping modes;
the eight-rank global shock crossings reproduce the serial diagnostics
at the printed precision, with zero inadmissible steps. The full moving
two-transit comparison above was measured in serial.

**Decision.** Retain the production default and subcycling as the supported
workaround. The normalized envelope is a viable benchmark-only candidate
for the fixed Cartesian layer: it removes that undershoot and preserves
the declared conservation and reflection budgets and lowers the finest-grid
error in the measured smooth case, but brings the global step's L2 order down to
the subcycled one and does not ensure species positivity after moving
refinement. The cadence comparator offers no demonstrated general
advantage over rate normalization and needs elapsed-time and phase handling
before it could become a production policy. This qualification is bounded
to the tested Float64 CPU cases; it does not identify an operator mode or
promote a general filter-interval multiplier.

### bench/archive/restrictcost.jl: restriction before every stage

`julia --project=. -t 8 bench/archive/restrictcost.jl study=time case=2d` (and `case=3d`),
`julia --project=. -t 1 bench/archive/restrictcost.jl study=shock`.

Restricting before every stage instead of once per step adds one to two percent to the step
and leaves a shock crossing nearly unchanged. The per-stage schedule injects the fine level
into the covered parent nodes before the shell imposition that follows every stage of the
global step, as `restrict=stage` of `bench/archive/temporalorder.jl` does. The cost rows alternate
the two schedules in one process, five repetitions of ten steps each, one rank at eight
threads, the default numerics on a smooth advected field; the ratio is the median of the
paired per-repetition ratios.

| case | restrictions per step | restriction, ms per step | share of the step | stage/step |
|---|---|---|---|---|
| 256² root, 97² box in 36 tiles, per step | 1.10 | 0.36 | 0.24% | |
| same, per stage | 7.00 | 2.39 | 1.53% | 1.021 |
| 64³ root, 31³ box in 8 tiles, per step | 1.10 | 1.57 | 0.20% | |
| same, per stage | 7.00 | 10.1 | 1.26% | 1.007 |

The Sod crossing of `bench/archive/interfacesensor.jl` gives the minimum density and pressure over
every step and the density error at t = 0.2 against the uniform run at the level-1 spacing:

```text
root N = 201, box over [0.6, 0.8]; at three levels a level-2 box over its middle half
cfl 0.4, the default numerics, global step
```

| levels, schedule | steps | ρ_min | p_min | density error |
|---|---|---|---|---|
| 2, per step | 707 | 0.06239 | 0.03822 | 8.145e-2 |
| 2, per stage | 707 | 0.06239 | 0.03822 | 8.201e-2 |
| 3, per step | 1761 | 0.05121 | 0.02508 | 8.298e-2 |
| 3, per stage | 1761 | 0.05121 | 0.02252 | 8.360e-2 |

**Decision.** The per-stage schedule costs six restrictions per step more, each a gather
over the parent level's communicator, and a measured 0.7–2.1% of the serial step. It changes
the smooth error against the exact solution by under 2%
([temporal order](walls.md#temporal-order)), raises the shock's density error by under 1%, and
deepens the three-level pressure minimum by a tenth. The once-per-step injection is retained
under both stepping modes.

### bench/archive/substeprates.jl: refreshed refined-level rates

`bench/archive/substeprates.jl` measures the rates after the refined RK right-hand
sides refresh the artificial coefficients, using the production subcycling
driver. The root estimate is taken after startup priming or regrid priming,
and the state filter uses the production cadence and relaxation.

At startup the refreshed substep CFL of a refined level reaches about
eight times the root target, in the first root step at both depths.
Reproduction: `julia --project=. -t 1 bench/archive/substeprates.jl steps=3
levels=3,4 regrid_steps=6`.

```text
serial, Float64, one Julia thread, Intel Core i9-12900K, Julia 1.11.4
201 root nodes, a Sod jump at x = 0.69 inside the deepest nested box, root CFL 0.2
```

| levels | peak refreshed substep CFL | divided by root CFL target | refined level | substep | RK stage |
|---|---|---|---|---|---|
| 3 | 1.607612 | 8.03806 | 1 | 3 | 5 |
| 4 | 1.704789 | 8.52395 | 1 | 2 | 1 |

The location is the level with the largest observed rate, not necessarily
the finest level. These are rate-growth measurements, not stability or
solution-accuracy certificates.

Under moving refinement the refreshed rates stay below the root target,
whether or not the layout has just changed. The supported two-level
moving-box case changes layout four times in six root steps. These readings
include the regrid coefficient refresh and therefore measure its resulting
trajectory.

| two-level moving box | CFL | over the root target | root step | refined level | substep | RK stage | box offset/extent |
|---|---|---|---|---|---|---|---|
| largest, changed layout | 0.188657 | 0.943283 | 2 | 1 | 6 | 1 | (50, 31) to (87, 30) |
| largest, unchanged layout | 0.198363 | | | | | | |

The proposed absolute ceiling of one was rejected: the subcycled Sod
precompile case at root CFL 0.2 reached a refreshed refined stage CFL of
1.011060 on its first step. The rate used to choose a step and the largest
rate within its RK trajectory are different quantities; that observation
alone does not establish a stability ceiling. The refreshed-rate
check therefore remains opt-in (`StepControl.substep_cfl = 0` by default),
with a positive value interpreted as an absolute ceiling qualified for the
case. A violation returns through the level communicators to the root's
savepoint/retry path before any accepted-step clock or callback advances.

Every regrid check refreshes artificial coefficients before the next
root estimate, including checks that keep the layout. This makes the refresh
cadence independent of timing-based ownership moves; a changed layout saves
the refreshed coefficients in its retry state. This applies to both direct
`regrid!` calls and the run-loop hook, for box and tiled refinement. Dynamic
regridding below the first refined level remains unsupported; the deeper
startup cases do not qualify that missing capability.

The `test/mpi_tests.jl` moving-Sod references are taken with this refresh
cadence, under round-off tolerances. Serial and two-rank runs agree to
round-off; the times follow from the fresh coefficients in the root CFL
estimate. All cases use 400 root nodes, root CFL 0.2 and the unrelaxed
filter.

| regression fixture | completed steps | serial time | final offset/extent or tile offsets |
|---|---|---|---|
| distributed regrid | 61 | 0.005505030961213709 | 159 / 43 |
| level rank subsets | 41 | 0.0055135979946853665 | 170 / 25 |
| tiled regrid | 41 | 0.003194481761204692 | 168, 176, 184 |
| stored ownership / rebalance | 41 | 0.003194481019080914 | 168, 176, 184 |

**Cost and dense-output decision.** The refreshed-rate guard's scan adds
about a tenth to a root step, and the endpoint evaluations of the dense
output are about a twentieth of the RHS time. Two warm whole runs precede
five pairs of three-step smooth runs, with the guard disabled and enabled on
identical initial data. The table reports the minimum time in each group;
differences between the two depth-dependent percentages are not resolved
against workstation timing noise.

| levels | guard off, ms/root step | guard on, ms/root step | scan overhead | estimated endpoint share of RHS time |
|---|---|---|---|---|
| 3 | 2.472 | 2.738 | 10.7% | 5.48% |
| 4 | 9.152 | 9.923 | 8.4% | 4.63% |

The endpoint estimate weights measured per-level RHS costs, the minimum of
five warm samples, by the recursive evaluation counts.

| levels | stage RHS evaluations | endpoint evaluations | RHS cost per level, μs |
|---|---|---|---|
| 3 | 65 | 4 | 39.8, 18.6, 26.0 |
| 4 | 200 | 13 | 32.3, 19.2, 27.0, 38.5 |

This is an estimate for the measured serial layouts, excluding endpoint box
gathers; it is neither an equal-cost count nor a distributed speedup bound.
Retain cubic Hermite output: removing this small RHS share does not justify
a new temporal reconstruction and its accuracy qualification. Revisit the
choice if a measured layout makes parent endpoint work dominant.

### bench/archive/amr_transfer.jl: the 3:1 transfer pair

`julia --project=. bench/archive/amr_transfer.jl` (`7dbf319`; banded schemes
`8271bd7`).

The transfer pair is exact to round-off as a round trip and converges at
the Lagrange order away from a closed end, and at the closure's third order
near one. The Pyranda pair, an invertible compact filter and a Gaussian
of width 3Δx, is bound to `plan_direction` as a `CompactScheme`
(restriction) and a `BandedCompactScheme` (prolongation). The public
kernels do not specify the 3:1 sampling convention, so it is pinned
numerically.

| quantity | measured |
|---|---|
| pair round-trip, closed / periodic | 1.6e-15 / 2.7e-15 |
| restriction as left inverse (coarse → fine → coarse) | 8.9e-16 |
| fine → coarse → fine order, Lagrange 4 / 6 / 8 | 3.97 / 5.93 / 7.97 |
| constant under restriction / prolongation | last bit / 13 ULP |
| round-trip order within 6 points of a closed end | 3 (the closure order) |
| plans against parity-extended full lines, both schemes, both signs | ≤ 1e-13 |
| prolongation gain at the fine Nyquist | 20.24 |
| closure condition number | 33 |

Conditioning in situ, on a captured shock:

| 2h captured shock | measured |
|---|---|
| smoothed δ⁴ sensor, round trip | 1.03–1.13 |
| state undershoot at the shock | ≤ 3% of ambient |
| pollution away from the shock | decays ≈ 3.4× per point |

On these figures the default coupling was judged tolerant of captured
features, and `tag_buffer = 4` was sized.

**Closure localization.** The response of the compact solve to a unit error in
the first ghost layer decays into the patch at exactly the root of the LHS
symbol.

| scheme | rate per point | response at 8 fine points | at 12 points |
|---|---|---|---|
| C6 | 0.382 | — | five orders down |
| C8 | 0.451 | — | — |
| C10 | 0.556 (1.8× per point) | 5e-3 | 5e-4 |

Settled: the default 4-coarse-cell tagging buffer holds two orders at C10
where it holds five at C6, which is enough; the buffers did not move.

### bench/archive/leveltransfer.jl: the live interpolation order

`julia --project=. -t 1 bench/archive/leveltransfer.jl orders=2,4,6,8,10` (about a
minute).

An interpolation of order p reaches that order in values, p − 1 in the
first derivative and p − 2 in the second, and regrids do not accumulate
transfer error. The study is the live coupling by itself, on analytic data
with the root exact: the `level_interpolation_order` Lagrange interpolation
that fills a refined patch's shell and a freshly created fine region, the
restriction, and repeated regrids.

```text
2-D fields     periodic on [0, 2π)², a square level fixed at 5L/12..7L/12
smooth field   1 + 0.2 sin(3x + 0.37) cos(2y − 0.1)
N              the root count
```

**Values.** A tensor polynomial of degree p − 1 per dimension is reproduced
in the shell and in a fresh fill to 1e-14 of its maximum, and one of degree
p is not:

| p | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| error on degree p, of its maximum | 7.5e-3 | 3.4e-4 | 4.7e-5 | 1.6e-5 | 1.3e-5 |

On the smooth field:

| order | ghosts, N = 48 / 96 / 192 | boundary planes, N = 192 | fresh fill, N = 192 | order |
|---|---|---|---|---|
| 2 | 4.86e-3 / 1.23e-3 / 3.01e-4 | 1.25e-4 | 3.09e-4 | 2.0 |
| 4 | 1.15e-4 / 7.28e-6 / 4.46e-7 | 2.23e-7 | 4.57e-7 | 4.0 |
| 6 | 3.28e-6 / 5.27e-8 / 8.09e-10 | 4.46e-10 | 8.30e-10 | 6.0 |
| 8 | 1.18e-7 / 4.66e-10 / 1.80e-12 | 9.37e-13 | 1.80e-12 | 8.0 |
| 10 | 6.70e-9 / 6.68e-12 / 6.66e-15 | 2.11e-15 | 6.66e-15 | 10.0 |

At order 8 the outermost ghost layer's stencil sits one node inward of
centred (`LEVEL_BUFFER = 4`), and at order 10 the outer two sit two and one
nodes inward; the ghost column reads the order regardless.

**Derivatives.** `interp` is the fine patch's operator on the imposed shell
minus the same operator on exact shell data, the transfer's own contribution;
`total` is against the exact derivative. The first derivative goes through
the gradient plans, whose extended interface rows read the ghosts; the
second applies the divergence plans (one-sided at the interface end, the
viscous flux path) to that first derivative, along x (xx) and along y (xy).
Windows are the four fine nodes nearest an end along the differentiated
dimension. The table gives the error at N = 192, with the orders over the
three grids in parentheses:

| order | 1st interp | 1st interp on a plane | xx interp | xy interp | xx total | fill, 1st interp |
|---|---|---|---|---|---|---|
| 2 | 3.66e-3 (1.0) | 1.62e-2 (1.0) | 0.61 (0.0) | 0.50 (0.0) | 0.61 | 2.21e-2 (1.0) |
| 4 | 6.54e-6 (3.0) | 2.39e-5 (3.0) | 3.77e-4 (2.0) | 6.52e-4 (2.0) | 3.76e-4 | 3.94e-5 (3.0) |
| 6 | 1.31e-8 (5.0) | 4.59e-8 (5.0) | 7.53e-7 (4.0) | 1.22e-6 (4.0) | 1.78e-6 (3.4) | 7.86e-8 (5.0) |
| 8 | 2.75e-11 (7.0) | 9.65e-11 (7.0) | 1.58e-9 (6.0) | 2.56e-9 (6.0) | 1.29e-6 (3.1) | 1.65e-10 (7.0) |
| 10 | 1.06e-13 (9.0, 8.4) | 2.27e-13 (9.0) | 8.01e-12 (8.0, 7.0) | 5.94e-12 (7.9) | 1.29e-6 (3.1) | 3.93e-13 (8.9) |

An O(h^p) value error enters the first derivative at h^(p−1) and the second
at h^(p−2), each at the fine spacing, as the counting predicts. The order-10
rows reach round-off on the finest grid. A fresh fill carries the
first-derivative loss over the whole patch, not only at its edge.

Under the default coupling the transfer of order 6 is visible on the
viscous path, where it is a large share of the second-derivative total. At
order 8 that total is the divergence's own closure error.

| order 6, xx | N = 96 | N = 192 |
|---|---|---|
| transfer term, share of the total | 64% | 42% |

**Restriction.** Injection writes the fine coincident values onto the
covered root nodes exactly. The filtered restriction is second order
against point samples, since it is not a point-sample transfer.

| restriction | N = 48 | N = 96 | N = 192 | order |
|---|---|---|---|---|
| injection, nodes written | 20 | 157 | 814 | |
| injection, difference | round-off | round-off | round-off | |
| filtered, error on point samples | 1.68e-3 | 4.63e-4 | 1.16e-4 | 1.9 / 2.0 |

**Repeated regrids.** Regrids do not accumulate transfer error. A
one-dimensional level is moved ±N/24 root nodes sixteen times with no
evolution, restricted after each move. The fine error after the last regrid
is 1.09 to 1.16 times the error after the first at every order and N, and
the order of both is the interpolation order.

| repeated regrids | order 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| measured order of the fine error | 2.0 | 4.0 | 6.0 | 8.0 | 9.9 |

| order 6, N = 192 | after the first regrid | after the last |
|---|---|---|
| fine error | 6.98e-10 | 7.63e-10 |

**Positivity.** Only order 2 is monotone on a fresh fill of a sharp step,
and the undershoot at the higher orders grows with the order. The fill is
of the step 1 → 0.125 of width w root cells. At w = 1 and 2 no order leaves
the range.

| fresh fill, w = 0.5, of the jump | order 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| undershoot of the low state | 1.2% | 2.3% | 2.9% | 3.2% |
| overshoot | 0.8% | 0.4% | 0.03% | 0.05% |

On resolved data the order makes no difference: the moving-region Sod gate
of `test/level_tests.jl` (N = 201, subcycled, regrid every five steps) is
order-independent.

| Sod gate, order | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| composite density error | 8.7e-4 | 8.4e-4 | 9.5e-4 | 8.9e-4 | 9.2e-4 |

| Sod gate, every order | |
|---|---|
| minimum density | 0.1243–0.1248 |
| minimum pressure | 0.0992–0.0995 |
| steps | 628 |

**Budgets.** Orders 8 and 10 keep every conservation figure inside its
budget. The run is `bench/interfaceconservation.jl`, serial,
`layouts=uniform,depth2`, otherwise the command of that section; the pairs
are global step / subcycled, the order in which the script runs them. The
order-8 figures were taken at `c7d26e3`, where the conservation section's
order-6 run read the same to two digits in every figure but the final drift.
The order-10 figures were recorded at `decf45a`. Both precede the
re-measurement of that section in the quadrature with the junction
correction Ω (`d346d35`), so they compare with each other and not with its
current figures.

| `interpolation_order` | 8 | 10 |
|---|---|---|
| two levels, drift (global / subcycled) | 6.95e-5 / 6.94e-5 | 6.88e-5 / 6.89e-5 |
| regrids | sixteen | eight |
| regrids, sum of absolute jumps | 8.41e-4 | 2.20e-4 |
| sampled drift with transfers | 1.0e-4 | |
| sampled drift, transfers subtracted | 2.1e-4 / 2.2e-4 | 1.20e-4 |
| final drift | 1.7e-5 / 2.5e-5 | 3.8e-6 / 4.5e-6 |
| species excursion | 7.4e-4 / 5.6e-4 | none outside [0, 1] |

Settled: the live order is a choice among the even orders 2 to 10, by
default the derivative operator's interior order, two more under
`interface_flux = :ghost`. Orders 8 and 10 fit the default halo and buffer,
the off-centre outer stencils included, cost nothing measurable in
conservation, positivity on resolved data or regrid drift, and remove the
transfer from the second-derivative and filtered error budgets. Order 2 is
the only monotone choice.

### test/patch_tests.jl: same-level patch interfaces

`julia --project=. test/runtests.jl` (the patch testsets). The rows under the
default ghost fluxes and the viscous rows are the testsets' output at
`743fdb4b` under Julia 1.13.3, and the drift is at `b4388f9`. The two rows reading "the same under
`:closure`" are the closure-row figures the test file records and no longer
runs (the C6 entropy orders are the `gflux` table's), and the row of the
ninth-degree polynomial is `test/serial/operators.jl`'s.

Settled: under the default ghost fluxes the inviscid orders through the
interface are the interior's (C6) or the time integrator's (C10), and
`interface_rhs = :onesided` is refused. Under `interface_flux = :closure` the
divergence's one-sided rows limit the inviscid orders, so
`interface_rhs = :extended` is measurable only through the gradients, where it
is worth two orders. C10 interface rows are the default halo's, no wider. The
cases are two conforming patches with manufactured smooth solutions across the
interface, since the bit-exact oracle does not survive it.

| gate | C6 | C10 |
|---|---|---|
| entropy-wave order across the interface | 6.72 / 5.89 | 4.00 / 3.97 |
| entropy-wave error, 48 / 96 / 192 | 8.7e-10, 8.2e-12, 1.4e-13 | 1.7e-11, 1.1e-12, 6.8e-14 |
| the same under `:closure` | 3.09 / 3.53 | 3.06 / 3.52 (8.2e-7, 9.8e-8, 8.5e-9) |
| acoustic pulse, reflected over incident amplitude at 192 | 8.2e-6 | 3.8e-7 |
| the same under `:closure` | 2.3e-3, order ≈ 5 | 4.1e-3 (7.5e-2 at 96, 7.7e-5 at 384) |
| viscous wave under `:closure`, `interface_rhs = :extended` | 4.34 / 4.00 | 3.93 / 3.92 |
| viscous errors at 48 / 96 / 192 | 6.1e-5, 3.0e-6, 1.9e-7 | 6.0e-5, 4.0e-6, 2.6e-7 |
| the same with `:onesided` | 1.92 / 1.97 (1.3e-4, 3.4e-5, 8.6e-6) | 2.14 / 1.93 (1.4e-4, 3.1e-5, 8.2e-6) |
| degree-9 polynomial through both closed ends | — | 2e-12, against 1e-4 for the cascade rows |
| conservation drift, long periodic run | 1.2e-11 relative (single patch 4.2e-15) | — |

Rank partitioning reproduces the serial two-patch answer bitwise at one rank
per patch. Once a patch itself decomposes, the agreement is round-off, with
identical step counts: at four ranks the difference is 3.1e-15 against a
9.5e-8 signal.

### test/level_tests.jl: the level hierarchy

`julia --project=. test/runtests.jl` (the level testsets); MPI legs through
`test/mpi_tests.jl`. Commits `772e304` and `e215c5a`, tiles `eb0cb38`,
covered masks `8d32ab3`, checkpoint `4b42724`.

**Coupling choice.** The default coupling is sixth-order Lagrange
prolongation with coincident-node injection, whose orders on a manufactured
solution are 3.46 / 3.64. The invertible pair as the live coupling reaches
only 1.3–1.7 at errors three decades higher, since prolongation's input must
be samples of the filtered field and the live coarse solution is not.

**Buffers.** The restriction margin breaks an amplifying loop under
`level_restriction = :filter` only. Restricting up to the face itself, the
C6 wave's error doubles every step and every filtered case fails within twenty
steps; at a margin of one node the error is flat. Under the default injection
no margin shows a loop: from t = 0.5 to 4 the wave error grows by under half a
percent per step, as an accumulating error does. There the two smallest
margins give the same wave
figures, since the node the smallest adds is the coincident node of the fine
boundary plane, which the shell imposes from the parent after every stage. A
margin of 1 lowers the wave error where it stands above round-off and lowers
the Sod noise, both against the default, but raises the Sod mass drift in
every configuration; a margin of 0 raises that drift under injection by one to
three orders. The constant stays at 2.

The restriction holds `RESTRICT_MARGIN` region nodes back from a parent-fed
face (`_restrict_window`). The derivative mask and the parent's overwritten
nodes are formed from the covered mask and do not read the constant, so the
sweep moves the restriction alone. It redefines `_restrict_window` in one
process, which at the default margin reproduces the constant bitwise, under
Julia 1.13.3 at `17c3ee3`. The wave is the two-level entropy wave of
`test/level_tests.jl` at N = 96, with the artificial properties and the filter
off, run to t = 4; its error is the largest density error over both patches.
Its gain per step is the geometric mean from step 6 to the failure or to step
24. The Sod crossing is the subcycled one of the same file, its noise the momentum
ahead of the shock at t = 0.1 and its drift the relative mass change at
t = 0.2. At `8271bd7` a `LEVEL_BUFFER` of 6 reproduced the default buffer's
figures to the last digit at both schemes.

| margin | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| wave error at t = 0.5, C6, injection, ghost fluxes | 1.89e-12 | 1.89e-12 | 1.95e-12 | 1.93e-12 |
| the same, closure rows | 4.97e-9 | 4.97e-9 | 7.68e-9 | 5.81e-9 |
| wave error at t = 0.5, C10, injection, ghost fluxes | 2.11e-14 | 2.11e-14 | 1.93e-14 | 2.04e-14 |
| the same, closure rows | 5.11e-9 | 5.11e-9 | 5.73e-9 | 5.41e-9 |
| wave error at t = 0.5, C6, `:filter`, ghost fluxes | fails at step 16 | 1.70e-5 | 3.06e-5 | 1.73e-5 |
| the same, closure rows | fails at step 19 | 1.75e-5 | 2.97e-5 | 1.74e-5 |
| wave error at t = 0.5, C10, `:filter`, ghost fluxes | fails at step 18 | 2.07e-5 | 2.80e-5 | 1.74e-5 |
| the same, closure rows | fails at step 19 | 1.97e-5 | 2.82e-5 | 1.73e-5 |
| wave error growth t = 0.5 → 2, C6 / C10, injection, ghost fluxes | 4.04 / 3.67 | 4.04 / 3.67 | 4.65 / 4.10 | 4.01 / 3.83 |
| the same, closure rows | 1.70 / 2.17 | 1.70 / 2.17 | 2.54 / 3.34 | 1.55 / 1.91 |
| the same, `:filter`, ghost fluxes | | 1.72 / 2.10 | 2.32 / 2.20 | 1.76 / 2.31 |
| the same, `:filter`, closure rows | | 1.73 / 2.13 | 2.25 / 2.19 | 1.78 / 2.32 |
| wave error gain per step from step 6, C6, `:filter`, closure rows / ghost fluxes | 2.02 / 2.25 | 1.00 / 1.00 | 1.01 / 1.01 | |
| Sod noise, C6 / C10, injection | 1.40e-10 / 3.15e-9 | 7.43e-11 / 1.96e-9 | 9.57e-11 / 3.44e-9 | 3.34e-10 / 2.04e-9 |
| Sod mass drift, C6 / C10, injection | −4.18e-5 / −8.52e-4 | −1.87e-6 / −1.08e-6 | −1.13e-6 / −7.82e-7 | −1.46e-6 / −5.69e-7 |
| Sod noise, C6 / C10, `:filter` | fails at step 17 | 6.59e-11 / 1.74e-9 | 8.99e-11 / 3.10e-9 | 3.02e-10 / 1.76e-9 |
| Sod mass drift, C6 / C10, `:filter` | | −1.12e-6 / −1.45e-6 | −2.25e-7 / −1.23e-6 | −1.53e-6 / −7.29e-7 |

**Subcycling.** Subcycling keeps the orders at a third of the steps, and on
the Sod gate it lowers both the noise ahead of the shock and the mass drift
under the coarse-fine correction.

| case | global step | subcycled |
|---|---|---|
| Sod gate, noise ahead of the shock | 3.5e-10 | 1.1e-10 |
| Sod gate, mass drift under the coarse-fine correction | 6.2e-8 | 2.3e-8 |
| C10 two-level wave, orders | 3.84 / 3.54 | 3.88 / 3.53 |
| C10 two-level wave, errors | 8.2e-8, 5.7e-9, 4.9e-10 | |
| C10 Sod crossing, momentum noise ahead of the shock | 5.4e-10 | 4.0e-10 |
| C6 Sod crossing, momentum noise ahead of the shock | 6.4e-10 | 1.3e-10 |

**Distributed coupling.** Distributing the interpolation chains by conserved
component, with the shell ring shared through one Allgatherv, brings the cost
case in three dimensions to 49% of the uniform-fine wall, from 85% with the
chain replicated on every rank. The serial results are bit-identical.

**Tiles.** A tiled level costs nothing visible in accuracy against the
one-patch level.

| case | tiled | one patch |
|---|---|---|
| 1-D entropy wave, tile 8, error at N = 192 | 6.0e-10 | 6.2e-10 |
| the same, orders | 3.95 / 3.91 | |
| 2-D nest of 2×2 tiles, corner included, error | 4.29e-8 | 4.27e-8 |

A flat pairwise pass over a node shared by four tiles does not end at the
mean of the four copies, which is why `_sync_level_records!` is
dimension-phased:

| copy | first | second | third | fourth |
|---|---|---|---|---|
| before the pass | 1 | 2 | 3 | 4 |
| after a flat pairwise pass | 2.23 | 2.68 | 2.41 | 2.68 |

**Moving regions.** A moving level puts the composite error about an order of
magnitude below the uniform coarse grid's: on Sod with fine resolution over a
third of the domain, and on Shu–Osher at about half the reference's steps. The
tiled regrid reproduces the Sod gate.

| case | composite | uniform coarse | improvement | coarse steps | reference steps |
|---|---|---|---|---|---|
| Sod, N = 201, density error against 601 uniform-fine nodes | 2.8e-3 | 7.3e-2 | 26× | | |
| Shu–Osher, L∞ over the wave train | | | 10× | 2497 | 4662 |
| Shu–Osher, L1 | | | 6.7× | | |

**Three regridded levels.** Three subcycled levels put the composite density
error at the root stations a hundred times below the root's alone, on a shock
driven through a contact into heavy gas. Decomposed runs at two, four and
eight ranks match the serial tile sets and density to 1e-14 to 7e-13.

| run: shock from x = 0.25 through a contact at x = 0.5, densities 1 / 0.125 / 0.5, N = 81, tile 8, subcycled, CFL 0.2, t = 0.25 | density L1 error at the root stations, against a 721-node uniform reference |
|---|---|
| `max_levels = 3` | 2.18e-4 |
| root alone | 2.36e-2 |

With the first discontinuity moved to x = 0.3 the case fails at its first
step at the table's CFL number, at two levels as at three, and runs at
CFL 0.1.

**Composite diagnostics.** Under the per-orthant covered mask the composite
energy history stays within 4.1e-8 of the single-level history in the
conserved quadrature, where the unmasked sum sits 1.3e-2 above it. Settled:
the per-orthant mask, against a per-dimension factor. The case is
Taylor–Green on a 24³ root with an 8³ region refined off-centre. The
histories cover 61 steps.

**Restart.** A restart reproduces the uninterrupted run at every slot, tag
history included, serially and decomposed:

| restart case | ranks | nodes | checkpoint step | final step |
|---|---|---|---|---|
| tiled and box Sod regrid | serial | | 23 | 130 |
| tiled Sod regrid | 2, 4, 8 | 400 | 21 | 41 |

A twelve-tile wave rebuilt from a six-tile initial region, written on half
the ranks and restored on all of them, continues to 1e-12 in the wave error.

**Axisymmetric levels.** Interpolating r·Q and dividing by r at the fine
nodes gives the window errors of interpolating Q to the second or third
figure. Settled: the conserved variables are interpolated. Without the
curvature terms in the ring flux the viscous window barely converges, and
under the closure rows the inviscid errors are larger and converge more
slowly. The instrument is the convergence rows' case `annulus_level_case`, its
fine window read against the uniform run at its spacing in equal steps at
t = 0.2, and an r-z drift case, with the variants built in a scratch copy of
the tree. The drift case makes the annulus periodic in z, with a pulse
crossing a square level of edge 12 to t = 0.4 and the filter and artificial
properties off.

| window error at N | 49 | 97 | 193 |
|---|---|---|---|
| Q, inviscid | 2.56e-7 | 2.86e-9 | 7.95e-11 |
| r·Q, inviscid | 2.59e-7 | 2.90e-9 | 7.92e-11 |
| Q, μ = 0.002 | 4.55e-7 | 9.26e-9 | 2.56e-10 |
| r·Q, μ = 0.002 | 4.56e-7 | 9.28e-9 | 2.56e-10 |
| Q, μ = 0.002, no curvature terms in the ring flux | 3.03e-6 | 2.86e-6 | 1.92e-6 |
| Cartesian, μ = 0.002 | 3.65e-7 | 7.43e-9 | 2.11e-10 |
| Q, closure rows, inviscid | 1.09e-6 | 5.89e-8 | 5.67e-9 |

The relative mass drift of the refined drift case stays within a fifth of the
root alone's on both metrics, and the ledger closes to round-off.

| relative mass drift, root nodes | 37 | 73 |
|---|---|---|
| r-z, refined | 8.73e-7 | 6.25e-8 |
| r-z, root alone | 8.42e-7 | 5.73e-8 |
| Cartesian, refined | 5.72e-7 | 5.68e-8 |
| Cartesian, root alone | 4.89e-7 | 4.78e-8 |

A standing wave touching the annulus walls was not used for the rows: its
data violate a compatibility condition of the curved wall, which caps its
order near two on any grid
([the slip wall of the r-z annulus](walls.md#the-slip-wall-of-the-r-z-annulus)).

**NSCBC faces on a level.** A level carrying an outflow face converges at
sixth order in its fine face window, and once the root resolves the pulse
its reflection is the fine run's. The instrument is the convergence row's
case `outflow_level_case`, an acoustic pulse carried by a stream out through
an `NSCBCOutflowBC` whose face a level carries, read against the uniform run
at the level's spacing in equal steps.

| N | 25 | 49 | 97 | 193 |
|---|---|---|---|---|
| density error over the fine face window at t = 0.35, halfway through the exit | 2.40e-5 | 3.52e-7 | 4.24e-9 | 4.61e-11 |
| successive order | | 6.09 | 6.37 | 6.52 |

The reflection is the largest |p − 1| left in the domain at t = 0.8, after
the pulse has gone:

| reflection at N | 25 | 49 | 97 |
|---|---|---|---|
| level at the face | 1.7364e-4 | 1.6053e-4 | 1.6051e-4 |
| uniform at the level's spacing | 1.6107e-4 | 1.6056e-4 | 1.6054e-4 |
| uniform at the root's spacing | 6.4315e-4 | 1.6652e-4 | 1.6067e-4 |

At N = 25 the root holds under three nodes per pulse width, and the level's
reflection carries that error. From N = 49 it is the fine run's to 2e-4 of
its value.

**Levels on the axis.** A level at the axis reproduces the uniform run on one
state and over one step. Unfiltered, its axis window departs from the uniform
run through a grid-scale growth of the axis itself: the cylindrical divergence
(1/r)D(r·) does not hold the π mode of ρu in its null space. A filter pass at
every step at the rows' strength removes the growth, and one at the default
strength does not at the finest spacing measured. The instrument is the axis
rows' case `axis_level_case`, the tile against the uniform run at its spacing
in equal steps, with scratch scripts. On one state the tile's right-hand side
agrees with the uniform run's at the axis to 1e-14, and one step from one
state agrees to 1e-17.

| axis window, unfiltered unless noted | t | errors |
|---|---|---|
| tile, N = 96, 192, 384 | 0.4 | 2.7e-4, 1.4e-5, 1.7e-6 |
| root alone at its own nodes, N = 96, 192, 384 | 0.4 | 1.1e-4, 8.4e-7, 1.3e-8 |
| the same pulse on the tile at a Cartesian plane, N = 96 | 0.4 | 9e-8 |
| tile, filtered at every step, αf = 0.45 | 0.3 | 4.6e-7, 3.0e-9, 3.0e-11 |
| tile, N = 768, filtered, αf = 0.45 | 0.4 | 1.4e-11 |
| tile, N = 768, filtered, the default αf = 0.47 | 0.4 | 3.9e-7 |

The unfiltered difference is a sawtooth emitted at the coarse-fine face whose
amplitude grows at the axis node. A uniform run without a level shows the
same response to a sawtooth in ρ, ρu and E near the axis, which grows by
nearly three orders of magnitude at the axis node and then decays. In ρ
alone, and in all three components at a Cartesian plane, the sawtooth does
not grow.

| sawtooth of 1e-9 over r < 0.3, uniform run | largest at node 1, t = 0.05 | t = 0.2 |
|---|---|---|
| in ρ, ρu and E | 6.0e-7 | 4.5e-8 |
| in ρ alone | stays at 1e-9 | |
| in ρ, ρu and E at a Cartesian plane | stays at 7e-10 | |

These measurements carry the radial pressure term in the area form. The
growth belongs to the unrefined axis, and it and the ∂p/∂r form that removes
it are measured under
[Grid-scale growth at the r-z axis](walls.md#grid-scale-growth-at-the-r-z-axis).

**Levels placed on a face by tags.** A level that tags place on a domain face
reproduces the uniform run at the level's spacing, and at an outflow its
reflection is the fine run's. On the slip wall the regrid's interpolated start
and the tile face between two tiles carry the difference. The instrument is
the `level_boundaries` testset of `test/level_tests.jl`, in equal steps
against the uniform run at the level's spacing.

| case, N = 49 | t | difference from the uniform run at the level's spacing |
|---|---|---|
| high slip wall of the standing wave, level placed by a regrid at step 0, one box | 0.2 | 4.0e-8 |
| the same, two tiles | 0.2 | 1.5e-7 |
| the same, level placed by the setup | 0.2 | 9.0e-9 |
| tiles following the acoustic pulse of the NSCBC case out through the outflow | 0.35 | 5.8e-7 |

| reflection of the NSCBC case at t = 0.8 | |
|---|---|
| tiles placed by tags | 1.6064e-4 |
| uniform at the fine spacing | 1.6056e-4 |
| uniform at the coarse spacing | 1.6652e-4 |

On a uniform state the tag sweep of a wall tile marks no node with its taps
clamped at the wall, and two with the face read as open, whose ghost layers
hold zeros.

**Levels regridded onto a fold and nested at one.** A level that a regrid
places on a symmetry plane or the r-z axis reproduces the uniform run at the
level's spacing, as does a second level nested at a plane once the state is
filtered. Unfiltered, the nest converges below its order: the ghost-flux
coupling of the nest carries a growing mode, a real eigenvalue of its one-step
map above one, which a filter pass every step or every tenth step removes.
The instrument is the fold testsets of `test/level_tests.jl`, in equal steps
against the uniform run at the level's spacing.

| case | N | t | one box | two tiles |
|---|---|---|---|---|
| standing wave, level placed on the low symmetry plane by a regrid at step 0 | 49 | 0.2 | 2.9e-8 | 1.8e-7 |
| the same on the r-z axis, filtered at every step | | | 3.6e-11 | 5.7e-11 |

A second level at either plane of the standing wave between two planes
reproduces the uniform run to round-off:

| standing wave between two planes, N = 48, t = 0.1 | second level at one plane | at the other |
|---|---|---|
| difference | 4.5e-13 | 2.3e-13 |

Unfiltered, the convergence row's case `plane_nest_case` at t = 0.4 reads the
second level against the uniform run on 9N nodes:

| N | 36 | 48 | 72 | order |
|---|---|---|---|---|
| plane window | 3.64e-11 | 1.06e-11 | 1.97e-12 | 4.2 |
| coarse-fine face window | 9.51e-11 | 3.10e-11 | 7.60e-12 | 3.6 |
| interior | 6.76e-11 | 2.23e-11 | 5.46e-12 | |
| face window, second level held inside the first (offset N/6) | 1.68e-10 | 5.78e-11 | 1.97e-11 | 3.1 |

The loss is a growing mode of the unfiltered coupling under the ghost fluxes.
`bench/archive/couplingspectrum.jl` linearizes the map over one step of 0.4/(18N)
about the state after the first and reads its eigenvalues of largest modulus.
The nest's leading eigenvalue is real, a mode that does not oscillate, whose
eigenvector peaks on the first level's node beside the restriction window and
carries an odd-even component on both levels. Its growth rate rises with N,
and neither the interpolation order nor a wider restriction margin removes
it. Under the closure rows the rate falls by more than an order of magnitude,
and a filter pass every step or every tenth step brings it to the uniform
run's, which is that of the nonlinear base state.

| map, growth rate of the leading eigenvalue per unit time | N | rate |
|---|---|---|
| uniform run | | 8.5e-4 |
| nest, ghost fluxes, real | 36 / 72 / 144 | 5.6 / 9.6 / 17.9 (e^2.2 to e^7.2 by t = 0.4) |
| one level at the plane (`plane_level_case`), oscillating pair | 36 / 72 | 0.51 / 2.42 |
| the same, real | 144 | 9.7 |
| nest, closure rows | 72 | 0.33 |
| one level at the plane, closure rows | 144 | 0.28 |
| nest, a filter pass every step or every tenth step | | 8.6e-4 |
| nest, interpolation order 6, 8 or 10 | 36 | 5.5 to 5.8 |
| nest, restriction margin of 4 / 6 parent nodes for the second level (a patched build) | | 4.1 / 3.6 |
| entropy wave through a nest of the same shape on a periodic line, real | 36 | 2.0 |

The mode's growth with the step count shows in the second level's plane
window. With the second level's restriction into the first switched off (a
patched build) the unfiltered window converges at sixth order, and with it on
the order falls as the steps at fixed N increase. Filtered at every step, the
convergence row's configuration, the face and interior windows converge at
6.24 and 6.08.

| second level's plane window, N = 36 / 48 / 72 | errors | order |
|---|---|---|
| unfiltered, restriction switched off | 6.9e-12, 1.3e-12, 1.3e-13 | 5.7 |
| unfiltered, restriction on, 6N steps | | 5.0 |
| the same, 18N steps | the table above | |
| the same, 54N steps | | 3.8 |
| filtered at every step | 5.6e-12, 1.2e-12, 1.2e-13 | |

The entropy wave through the periodic nest converges at 7.1 to 7.7 with the
second level's restriction on or off, although its map carries a real
eigenvalue above one: its error excites the mode too little to show it, as in
the three-level entropy rows.

**Levels across a periodic seam.** A level across the periodic seam computes
the solution of a level clear of it to round-off, and its interface window
converges at sixth order. The whole composite mass drift of a box across the
seam is the junction's flux mismatch. The convergence row's case
`seam_level_case` is read against the uniform run at the level's spacing in
2N equal steps to t = 0.5:

| N | 48 | 96 | 192 |
|---|---|---|---|
| density error, fine interface window | 1.370e-7 | 1.727e-9 | 2.414e-11 |
| successive order | | 6.31 | 6.16 |
| density error, interior window | 4.44e-8 | 8.18e-10 | 1.54e-11 |

The seam testset of `test/level_tests.jl` translates a periodic run so that
the level lies across the seam in one run and clear of it in the other. The
two runs differ over every patch by:

| translated case | shift, root nodes | steps | difference |
|---|---|---|---|
| box in 1-D under either stepping mode, and corner box across both seams in 2-D | 36 | 40 | 1.1e-14 |
| two tiles abutting across the seam, lattice edge 6, N = 48 | 30 | | 8.4e-15 |
| level at the axis of an r-z run across the seam in z, static or regridded onto the axis, box | | 8 | 2.2e-14 |
| the same, tiles | | 8 | 1.9e-14 |

The composite mass drift of the corner box is the same on both sides of the
seam to 1e-15, and the ledger's coarse-fine and covered-face columns account
for all of it:

| corner box in 2-D, 12 steps | across the seam and clear of it |
|---|---|
| composite mass drift | −5.92193e-10 |
| sum of the ledger's coarse-fine and covered-face columns | −5.92194e-10 |

A density step through a box across the seam, which the coarse-fine
correction acts on, matches its translation to 4.4e-14 and conserves mass to
4e-15. A level regridded across the seam under `level_boundaries`, and the
entropy wave's composite mass with a level on either side of the seam:

| case | measure | value |
|---|---|---|
| box regridded across the seam, following a window at unit speed, 96 steps | difference from the uniform fine run | 4.8e-10 |
| the same, tiles | difference from the uniform fine run | 4.3e-10 |
| entropy wave, N = 48, t = 0.5, level clear of the seam | composite mass drift | −3.9e-11 |
| the same, level across the seam | composite mass drift | +1.8e-11 |

**A converging shock on the axis level.** A tile holding the shock's whole
path reproduces the uniform fine run on the axis as at a symmetry plane. A
shorter tile, which the shock leaves, loses mass at the crossing and
post-shock plateau inside the tile, at the axis as at the plane, so the loss
is the coupling's at a shock and not the axis's. The conservative coupling holds the crossing's
mass but not its plateau. The instrument is `noh_axis_level`
(`test/cases.jl`): cylindrical Noh from the cold start at N = 256 and CFL 0.15
with the default filter. A static tile over the first m root nodes is read
against the uniform run on 767 nodes. The global-step rows at m = 85 and 43
are guarded in `test/validation.jl`, whose header carries them. The table
holds scratch variants of the same case and a planar control (folded Noh,
N = 400, a tile at the symmetry plane over the same fractions of the line,
against the uniform run on 1199 nodes), the axis rows with the radial
pressure term in the area form:

| run | plateau | deficit | shock | mass drift | max \|ρ − uniform\| on the tile | steps |
|---|---|---|---|---|---|---|
| axis, m = 86, subcycled | 15.6524 | 48.3% | 0.2033 | −6.0e-7 | 0.27 | 3506 |
| axis, m = 43, subcycled | 15.099 | 56.8% | 0.2079 | −3.2e-3 | 9.9 | 3320 |
| plane, uniform on 1199 nodes | 3.9989 | 22.7% | 0.2008 | +1e-13 | | 10487 |
| plane, m = 134 | 3.9989 | 22.7% | 0.2008 | −2e-12 | 1.5e-4 | 10487 |
| plane, m = 67 | 3.9507 | 22.3% | 0.2036 | −1.14e-3 | 2.0 | 9739 |

The table's rows are uncorrected (`conservative = false`). The subcycled tile's
axis node reads a deficit 1.7 points above the global step's. Once the shock
leaves the tile, near t = 0.5, the uncorrected composite loses mass at the
crossing and the post-shock plateau inside the tile falls. On the axis it
falls to within 1.5% of the battery's uniform cylindrical row at the root's
resolution, and at the plane by 1.2%. Under the global step the conservative
coupling
([bench/archive/reflux.jl](#bencharchiverefluxjl-the-conservative-coarse-fine-coupling))
leaves the axis crossing's plateau where the uncorrected run puts it:

| axis crossing, global step | mass drift in the conserved quadrature | plateau |
|---|---|---|
| conservative coupling | −3.6e-7 | 14.906 |
| uncorrected | | 14.912 |

No axis runaway appears under αf = 0.47 at the level's spacing, 767 radial
nodes.

**Reproducibility tier.** A tile owned by a proper subset of the ranks
reproduces the every-rank answer to round-off, not bitwise.

| case, at 2, 4 and 8 ranks | difference |
|---|---|
| tiled wave cases, a tile owned by a proper subset, against every rank | 0 to 6e-15 |
| tiled Sod regrid with rebalancing on, its final time against the serial one | 5e-18 |
| a decomposed patch | 1e-15 to 6e-15 |

### bench/archive/amr_tiles.jl: tiled cover and per-tile cost

`julia --project=. bench/archive/amr_tiles.jl [N=192] [tile=6] [steps=3]` (`eb0cb38`;
workspace pooling `41d212f`, patch-type compile `556d69a`).

Settled: the lattice cover is worth having. A tiled lattice covers an annular
tag set with under half of its bounding box, and the ratio improves with a
thinner shell, which is the implosion argument for tiles over one box.

| tile edge (annular tag set, N = 192, r0 = 0.75, w = 0.02, buffer 1) | cover of the bounding box | tiles | fine nodes each | memory each |
|---|---|---|---|---|
| 6 | 41% | 208 | 19² | 0.40 MB |
| 12 | 47% | 80 | 37² | 1.1 MB |

Per-tile cost: setup time lies in plan construction and grows with the tile
count, which argues for tile edges of 12 or more in 3-D. Native code compiles
once per distinct `Patch` type, and `promote_typeof` specialization cost time
per distinct patch count, which is why every refined face carries
`InterfaceBC` and `solver.patches` is a typed vector.

| setup cost | time |
|---|---|
| plan construction, N = 96, 64 tiles | 7.6 s |
| the same, 196 tiles | 11.4 s |
| warm construction of plans, scratch, transfers and communicators, eight tiles | 0.03 s |
| native code per distinct `Patch` type, CPU backend | 1.8 s |
| the same, device backend | 3.9 s |
| `promote_typeof` specialization per distinct patch count | 0.5–1.7 s |

Pooling the right-hand-side workspace cuts the memory of the patch set by a
factor of 1.71, since about half of each tile's memory is scratch a pool can
share. The warm step does not change outside the run-to-run spread.

| tile 6: 208 tiles of 19² plus the root, one rank on 16 threads | before pooling | after |
|---|---|---|
| memory over the patch set | 103.0 MB | 60.3 MB |
| memory of each tile | 0.398 MB | |
| of it shared after pooling | | 0.207 MB |
| warm step | 0.55 s | 0.56 s |

Workstation pathology: at eight ranks any two-dimensional case, one patch or
four tiles alike, runs at ~7 s/step against ~0.5 s at four ranks, of the kind
`CLUSTER.md` records for hybrid cores. The MPI suite's tiled check is bounded
to ten steps for it.

Open: one cold measurement of a configuration of large tiles took four to five
times a rough count of its right-hand-side work, while the warm annular
reading at a different configuration matches expectation. Remeasure warm at
the original configuration.

| configuration, per subcycled step | rough count of the RHS work | measured |
|---|---|---|
| 40 tiles of 37² on a 96² root | near 2 s | 8–10 s, one cold run |

### bench/rmicost.jl: whole tiles, the subdivided lattice and per-dimension edges

`mpiexec -n 8 julia --project=. -t 1 bench/rmicost.jl config=amr2 dim=3
affinity=0x5555` and the other legs below, on `98031e2` (base: tiles split
over rank ranges) against the same tree with whole-tile ownership and the
lattice subdivision; P-cores, the two trees alternated in one session. The
columns give ms per root step over the steady steps (mean, median), steady
wall per simulated µs, busy max/mean, the waiting share of the summed steady
wall, refined node-steps per level, and the mix width W, whose values on the
uniform fine runs are:

| uniform fine run | W mm |
|---|---|
| 2-D fine2 | 6.239 |
| 2-D fine3 | 6.075 |
| 3-D fine2 | 5.608 |

The amr3 rows in three dimensions cover the first 17 root steps, 7 of them
steady.

| leg | tree | lattice at the end | ms/step | median | s/µs | busy | wait | node-steps L1 / L2 | W mm |
|---|---|---|---|---|---|---|---|---|---|
| 2-D amr2 | base | 12 | 16.6, 19.2 | 14.3, 15.1 | 0.0090, 0.0104 | 1.09 | 18–20% | 3.54e6 | 6.2597 |
| 2-D amr2 | new | (6, 6) | 16.0, 19.7 | 8.9, 9.5 | 0.0088, 0.0108 | 1.03 | 21–26% | 3.06e6 | 6.2720 |
| 2-D amr2 | `tile=4,12` | (4, 6) | 16.0, 26.5 | 10.0, 10.6 | 0.0088, 0.0145 | 1.03–1.09 | 26–36% | 2.89e6 | 6.2856 |
| 2-D amr3 | base | 12 / 12 | 145.9, 149.9 | 109.6, 114.1 | 0.092, 0.095 | 1.10 | 33–35% | 5.41e6 / 7.67e7 | 6.0893 |
| 2-D amr3 | new | (6, 6) / 12 | 148.1, 142.2 | 113.1, 107.4 | 0.094, 0.090 | 1.06 | 37–39% | 5.36e6 / 7.52e7 | 6.0835 |
| 2-D amr3 | `tile=4,12` | (4, 6) / (4, 12) | 119.6 | 92.5 | 0.075 | 1.07 | 41% | 4.86e6 / 3.48e7 | 6.0681 |
| 3-D amr2 | base | 12 | 472.6 | 458.4 | 0.351 | 1.11 | 18% | 1.77e8 | 5.6358 |
| 3-D amr2 | new | (6, 6, 12) | 401.5 | 414.9 | 0.301 | 1.05 | 21% | 1.53e8 | 5.6454 |
| 3-D amr2 | `tile=4,12,12` | (4, 6, 12) | 410.9 | 433.0 | 0.305 | 1.06 | 28% | 1.40e8 | 5.6599 |
| 3-D amr3, 17 steps | base | 12 / 12 | 15970 | 15310 | 10.9 | 1.06 | 21% | 9.2e6 / 2.94e8 | |
| 3-D amr3, 17 steps | new | (6, 6, 12) / 12 | 18580 | 18160 | 12.2 | 1.04 | 23% | 9.4e6 / 3.33e8 | |

Settled: whole tiles on the subdivided lattice take 15% off the step of the
three-dimensional amr2 leg and bring its busy max/mean close to one. There the
base had split each of its first level's few large tiles over two ranks:

| 3-D amr2, first level | tiles | fine nodes each | ranks per tile |
|---|---|---|---|
| base | 4 or 5 | 37 × 36 × 36 | 2 |
| new, subdivided lattice | 12 to 28 | | 1 |

Thin tiles along the shock normal (`tile = (4, 12, 12)`) cover the planar
fronts with fewer refined node-steps than cubes. In 2-D amr3 they halve the
second level's node-steps and take 18% off the step. In 3-D amr2 the saving
in nodes is spent on the extra tile faces, and the step is the subdivided
cube lattice's. In 2-D amr2 the median step falls by a third and the mean
does not move, because the regrid checks rebuild more small tiles than the
base did:

| 2-D amr2, regrid checks | tiles rebuilt per check | share of the steady wall |
|---|---|---|
| base | 2 to 5 | 8% |
| new | 14 to 16 | 45% |

In two dimensions the second level of amr3 already holds 12 to 30 tiles on
the configured lattice and is not subdivided, and its mean step moves inside
the spread.

A sampling profile of the first 200 root steps of 3-D amr2 (`profile=true
nmax=200`) places the cost of whole tiles in the ghost-flux divergence, whose
passes run per tile face and double with the tile count. The other phases are
cheaper, and the level's artificial coefficients and right-hand side spend a
smaller share of their time inside MPI.jl calls.

| 3-D amr2, ms per root step (share inside MPI.jl calls) | base | new |
|---|---|---|
| level artificial coefficients | 153.6 (39%) | 101.9 (22%) |
| right-hand side | 107.0 (31%) | 74.1 (14%) |
| same-level sync | 41.4 (60%) | 33.1 (78%) |
| ghost-flux divergence | 50.4 (47%) | 98.0 (15%) |
| step | 422.1 (39%) | 374.1 (23%) |

W moves at the third or fourth figure with the layout: the subdivided and the
thin lattices cover the features more tightly, so less of the flow is at the
fine spacing, and W lies further from the uniform fine run's. The serial
runs, the convergence rows and the MPI suite's serial comparisons are
unchanged, since one rank never subdivides and a whole tile computes as the
serial tile does.

#### The lattice chosen by the measured cost model

The same legs on `a02b946` (base: the coarsest lattice giving a tile per
rank) against the lattice of the shortest predicted step (`_choose_lattice`),
the model's coefficients measured on the run; the new tree's script against
both projects, which adds the compilation after the warm-up, the most of
any rank. The coefficients at the end of each run, in seconds per step of the
level, but c per check:

| leg | a, per node | b, per node and interface dimension | k, per tile | m, per message | c, per tile at a check |
|---|---|---|---|---|---|
| 2-D amr2 | 1.2e-6 | 2.1e-7 | 3.3e-4 | 1–3e-4 | 7.3e-4 |
| 3-D amr2 | 2.3e-6 | 3.2e-7 | 0.024 | 4.0e-3 | 5.1e-3 |
| 3-D amr3 | 3.4e-6 | 3.4e-7 | 6.9e-4 | 5.1e-3 | 0.026 |

| leg | tree | lattice at the end | ms/step | median | compiling s | s/µs | busy | node-steps L1 / L2 | W mm |
|---|---|---|---|---|---|---|---|---|---|
| 2-D amr2 | base | (6, 6) | 16.5, 15.8 | 8.88, 8.89 | 3.5, 3.5 | 0.0090, 0.0086 | 1.04 | 3.06e6 | 6.2720 |
| 2-D amr2 | model | (6, 12) | 84.1, 83.9 | 7.26, 7.30 | 9.1, 9.0 | 0.046, 0.046 | 1.45 | 3.06e6 | 6.2700 |
| 2-D amr3 | base | (6, 6) / 12 | 138.7 | 104.3 | 5.5 | 0.088 | 1.07 | 5.36e6 / 7.52e7 | 6.0835 |
| 2-D amr3 | model | (6, 12) / 12 | 188.9 | 108.4 | 12.2 | 0.120 | 1.27 | 5.18e6 / 7.66e7 | 6.0898 |
| 3-D amr2 | base | (6, 6, 12) | 399.6 | 398.7 | 5.8 | 0.299 | 1.05 | 1.53e8 | 5.6454 |
| 3-D amr2 | model | (6, 12, 12), 8 tiles of 24624 nodes | 347.0 | 318.0 | 5.5 | 0.261 | 1.07 | 1.48e8 | 5.6452 |
| 3-D amr3, 17 steps | base | (6, 6, 12) / 12 | 15540 | 15200 | 5.7 | 10.18 | 1.04 | 9.41e6 / 3.33e8 | |
| 3-D amr3, 17 steps | model | (6, 6, 12) / 12 | 14230 | 14230 | 5.9 | 9.72 | 1.04 | 8.81e6 / 3.00e8 | |

Settled: in three dimensions the model's lattice is the faster. On amr2 it
takes the half-subdivided lattice, a tile per rank, over the lattice the
count rule reaches, and the step falls 13%. On amr3 the first level ends on
the count rule's lattice, chosen because a level of whole tiles spans as many
ranks as it has tiles and its children are placed within them. On the
configured lattice the first level would hold four tiles and leave the
second, nine tenths of the work, on four ranks. The window runs 8% faster
than the base with 10% fewer node-steps on the second level, whose base count
is that of the `98031e2` table above. The level's share of the step alone
would have kept the first level coarse.

Open: in two dimensions the model's lattice raises the mean step while the
median falls or holds, and all of the rise is in the regrid checks. A traced
run (`trace=true`) puts a check that places a tile of a face combination new
to its rank at 1.3 to 3.2 s of wall, nearly all of it one rank compiling the
step's tile phases while the others wait at the check's collectives. Less
that compilation, the checks cost about 1 ms per tile.

| 2-D leg, model against base | median step | mean step |
|---|---|---|
| amr2 | 18% lower | five-fold |
| amr3 | +4% | 1.4-fold |

| 2-D amr2 | wall of the regrid checks over the median step |
|---|---|
| base | 1.6–1.8 s |
| model | 17.9 s |

The count rule subdivided at the first check and met its combinations in the
first ten steps on every rank at once. The model starts on the configured
lattice, changes it once an interval is measured, and the tile set then grows
rank by rank, so the compilations come one at a time inside the steady
window. The compiling column gives the compilation of the most compiling rank
after the warm-up, inside the steps or between them (Julia's process-wide
counter). The model takes a check's wall less the longest compilation in it,
and leaves out the observations of a rank that compiled in an interval.
Without that exclusion it swung between the configured lattice and the
half-subdivided one on compile spikes.

### bench/archive/latticescale.jl: the lattice decision at a cluster's rank count

`julia --project=. -t 1 bench/archive/latticescale.jl affinity=0x5555`, serial on
one P-core, on synthetic three-dimensional Richtmyer–Meshkov fronts at 48³
root nodes per rank and a configured edge of 16. The table gives seconds of
each step of one regrid check's decision, which every rank (or rank 0, for a
deeper level) runs. Base is `593f673` (`_lattice_wanted` and `_tile_faces`
pairwise, the knapsack's proximity over every rank); new is the tree that
indexes the old tiles and bounds the chain at `_lattice_budget`. In the columns, "price" is
the face table, the placement and the layout cost of every lattice of the
chain, "inherit" the creation checks of a re-chopped level, "nest" the second
level's nesting test (on rank 0), and "place" the stored-ownership placement
and the balance check.

| P | tree | lattices | tiles, finest | wanted | price | inherit | place | nest |
|---|---|---|---|---|---|---|---|---|
| 112 | base | 7 | 25728 | 20.2 | 6.6 | 3.7 | 0.000 | |
| 112 | new | 7 | 25728 | 0.013 | 0.17 | 0.012 | 0.001 | 0.011 |
| 1120 | base | 5 of 6 | 43321 | > 300 | 20.3 | 33.2 | | |
| 1120 | new | 6 | 82956 | 0.057 | 0.51 | 0.071 | 0.009 | 0.043 |
| 11200 | new | 4 | 153035 | 0.093 | 0.64 | 0.17 | 0.25 | 0.38 |

Settled: the base decision was quadratic in the tiles (each cell of every
lattice tested against every old tile, every pair of tiles in the face
table), took 30 s a check at one node and never finished a check at ten.
The new one is linear in the tiles and costs about 1 s a check at 11200
ranks. That is small against a predicted root step of 13 to 160 s on these
fronts, with a check every four steps. The budget stops the chain at 4.6 MB
of cell flags at 11200 ranks, where halving to four parent nodes would reach
tens of megabytes on every rank. The fold of the gathered timings takes
milliseconds.

The script's second part prices four rules for two refined levels under the
model, with nodes of 112 ranks and the coefficients fitted on the workstation
(3-D amr2 at eight ranks, below). The cost of a message leaving the node
(m_x = ratio m) has not been measured and is swept.

| coefficients, fitted on 3-D amr2 at eight ranks | a | b | k | m | c |
|---|---|---|---|---|---|
| s | 2.15e-6 | 3.2e-7 | 0.028 | 5.4e-3 | 0.044 |

Every entry is the model's prediction, s per root step, with the best rule's
step in the last column:

| scene | P | ratio | configured | count rule | model, one node | model, nodes | best |
|---|---|---|---|---|---|---|---|
| thin | 11200 | 1 | 20.9 | 12.9 | 12.9 | 13.5 | 12.9 |
| thin | 11200 | 10 | 31.1 | 20.9 | 20.9 | 18.3 | 18.3 |
| thin | 1120 | 1 | 29.0 | 29.0 | 28.2 | 29.5 | 28.2 |
| thin | 1120 | 10 | 41.8 | 41.8 | 41.4 | 36.5 | 36.5 |
| front | 1120 | 10 | 185.3 | 185.3 | 187.0 | 143.3 | 143.3 |
| front | 11200 | 10 | 159.3 | 159.3 | 159.3 | 127.8 | 127.8 |

Modeled, not measured: early in a run (thin) at 11200 ranks the configured
lattice gives the first level 0.64 tiles per rank and confines the second
level to those ranks. The subdivided lattice the count rule and the model
take is predicted 1.6 times faster. Where the off-node message costs ten
times one within a node, placing a level's tiles by node takes 12 to 23% off
the predicted step. Where it costs the same, placement by node is predicted
4 to 5% slower than the single-node knapsack, since the runs cut per node
balance a little worse, so the gain rests on the ratio rzhound measures
(bench/archive/slurm/rmicost_lattice.sbatch, not yet run).

The workstation smoke at eight ranks (bench/rmicost.jl, P-cores, two runs a
tree alternated, base `593f673`) shows the new decision at the base's step in
three dimensions, on the same lattice and at the same W. In two dimensions
the new tree's median step and steady wall are lower, the difference lying in
the regrid checks and not resolved further. A lattice is three edges in every
dimensionality; in two dimensions the third is the collapsed dimension's
configured edge, which the subdivision leaves alone.

| leg | tree | lattice at the end | ms per root step | median ms | steady wall s | regrid checks s | W mm |
|---|---|---|---|---|---|---|---|
| 3-D amr2 | base | (6, 12, 12) | 342.5, 340.7 | | | | 5.6452 |
| 3-D amr2 | new | (6, 12, 12) | 342.6, 342.1 | | | | 5.6452 |
| 2-D amr2 | base | (6, 12, 12) | | 8.70, 8.04 | 15.2, 10.6 | 12.5, 8.5 | |
| 2-D amr2 | new | (6, 12, 12) | | 7.35, 7.33 | 5.9, 5.9 | 3.9 each | |

The fitted c, per tile of the busiest rank, is larger in three dimensions
than in two:

| fitted c per tile of the busiest rank | 3-D | 2-D |
|---|---|---|
| s | 0.036 to 0.051 | 0.0012 to 0.0125 |

In the MPI suite's measured three-level shock on fake nodes of two ranks, the
fit separated m_x (3.9e-5 s) from m (9.4e-5 s) in one run of four and fitted
one coefficient in the others.

### bench/archive/amr_cost.jl: the mixing cost case

`mpiexec -n 8 julia --project=. -t 1 bench/archive/amr_cost.jl 48 1.0` (`6382d8e`;
masked quadrature and tag criteria `f8e1a23`).

Settled: the composite buys most of the fine answer at a third to a half of
its cost, and the artificial diffusivity number alone tracks the blob to the
same final region. The case is a heavy-gas blob mixing on a 48³ root with a
subcycled, regridding region covering a sixth of the volume, run to t = 1
against the uniform coarse and fine grids of the table. The metric is
∫Y(1−Y)dV on the shared coarse lattice.

| configuration | mixedness error | wall | memory |
|---|---|---|---|
| coarse 48³ | 6.9e-3 | 18 s | 179 MiB |
| composite, δ⁴ρ tag | 1.5e-3 (4.6× closer) | 204 s (43% of fine) | 655 MiB (24% of fine) |
| composite, sensor tag (`tag=sensor sensor=0.02`) | 1.7e-3 | 35% of fine | — |
| fine 142³ | reference | 471 s | 2737 MiB |

Single runs at np = 8; read the ratios, not the third digit. The sensor tag's
threshold of 0.02 compares with about 2 on a captured Sod shock under the
default C_β. On the coarse grid no node farther than 0.6 from the interface
reads above the threshold.

Dead end: pointwise in-region error is the wrong metric. Coarse and composite
both sit at max ≈ 0.19 against fine there, the sub-cell displacement of a
near-discontinuous interface.

### bench/amrwin.jl: time to solution against the uniform fine grid

`julia --project=. -t 1 bench/amrwin.jl problems=<problem>`, one problem per
process, one rank and one thread on the i9-12900K. The tables of walls, steps
and errors and the root-step account were taken under Julia 1.13.3 at the
commit that records them. The overhead profile and the per-point timings
further down are older, taken under Julia 1.11.4 before the root-step change
described here. Each problem runs warm as the root grid alone (coarse), the
root with one subcycled regridded box, the same with lattice tiles, and the
uniform grid at the refined spacing (fine). The error is the composite L1
norm of ρ against the fine run at coincident nodes. Single runs; read the
ratios to two figures.

A refined level takes the fine run's step count on every problem except under
the box of the 3-D blob, where it takes a quarter more. The Sod tube and that
box, whose levels hold about a quarter of the fine grid's points, run well
below the fine wall. The levels of the other runs hold from a third to nine
tenths of the fine grid's points, at a cost per point-step between a third
above the fine run's and nearly three times it, and those runs take from a few
percent to a quarter longer than the fine grid.

| problem | root / fine grid | end time | fine wall | fine steps |
|---|---|---|---|---|
| sod | 201 × 17 / 601 × 49, slip walls, cfl 0.2, tile 8 | 0.1 | 17.1 s | 1347 |
| blobs | 64² / 192², periodic, edge 0.01 (0.64 root spacings), tile 16 | 0.2 | 5.34 s | 213 |
| shock | 256 / 767, r-z axis, tile 16 | 0.221 | 0.88 s | 2023 |
| blob3d | 24³ / 72³, periodic, tile 8 | 1.0 | 54.8 s | 104 |

The wall ratio is the product of the point-step ratio and the ratio of the
wall per point-step, each against the fine run. The level's steps count every
substep, and its points are the time average of the level-1 nodes.

| problem | run | wall / fine | point-steps / fine | per point-step / fine | level steps / fine | level points / fine | L1 ρ |
|---|---|---|---|---|---|---|---|
| sod | coarse | 0.046 | 0.041 | 1.12 | | | 2.6e-3 |
| sod | box | 0.329 | 0.240 | 1.37 | 0.996 | 0.203 | 3.75e-5 |
| sod | tiles | 0.522 | 0.284 | 1.84 | 0.996 | 0.247 | 6.79e-5 |
| blobs | coarse | 0.055 | 0.047 | 1.17 | | | 4.5e-2 |
| blobs | box | 1.04 | 0.776 | 1.34 | 1.01 | 0.73 | 3.90e-5 |
| blobs | tiles | 1.27 | 0.867 | 1.46 | 1.01 | 0.82 | 1.75e-4 |
| shock | coarse | 0.118 | 0.111 | 1.06 | | | 1.6e-2 |
| shock | box | 1.04 | 0.523 | 2.00 | 0.997 | 0.41 | 8.02e-4 |
| shock | tiles | 1.18 | 0.418 | 2.82 | 0.994 | 0.31 | 7.81e-4 |
| blob3d | coarse | 0.016 | 0.016 | 0.96 | | | 4.9e-3 |
| blob3d | box | 0.576 | 0.339 | 1.70 | 1.24 | 0.26 | 9.26e-5 |
| blob3d | tiles | 1.23 | 0.930 | 1.32 | 1.01 | 0.91 | 5.29e-5 |

Every run completes under the default validity policy. The shock's runs take
about a second, short enough that a single process's wall moves its ratios.

The mixedness of both blob problems stays within a percent of the fine run's
under the box and under the tiles:

| mixedness ∫Y(1−Y)dV against the fine run's | coarse | box | tiles |
|---|---|---|---|
| blobs | +122% | −0.004% | +0.34% |
| blob3d | +114% | −0.54% | +0.51% |

**The root step.** A parent's covered nodes carry the restricted fine
solution, in which a feature the level resolves is a jump on the parent's
spacing. The parent's sensors read that jump at full strength. The
coefficients built from them bounded most root steps of the refined runs,
whose levels took up to two and a half times the fine run's steps, and the
smoother carried them to the nodes beside a coarse-fine face, which the
parent keeps. The parent's sensors are now set to zero at the nodes the
restriction overwrites, before they are smoothed (`MASK_CHILD_SENSORS`).

Four forms were measured against the package without a mask, at `02ba4294`,
one process each. They set to zero either the coefficients, after smoothing,
or the sensors, before it, and they do so either on the nodes covered over
their whole cells or on the nodes the restriction writes, which leave the
first node inside a face to the parent. The adopted form is the last one.
Only that form completes every run with no error above the unmasked
package's, and its coefficients are continuous across the first masked node.
Its mass defect where a shock passes out of a level is the largest of the
three measured, and is the remainder of a correction a hundred times larger.

| form | blobs box: root steps | L1 ρ | Sod tiles: Linf ρ | shock box: Linf ρ | 3-D blob box: root steps | L1 ρ | Linf ρ | Noh leaving a level: mass |
|---|---|---|---|---|---|---|---|---|
| no mask | 96 | 2.52e-4 | 1.17e-3 | fails at step 677 | fails at step 12 | | | +6.7e-6 |
| coefficients, covered whole | 72 | 3.46e-5 | 1.27e-3 | 0.240 | 56 | 5.18e-5 | 5.76e-3 | −3.4e-6 |
| coefficients, written | 73 | 4.91e-5 | 1.15e-3 | 0.408 | 63 | 2.29e-4 | 3.23e-2 | |
| sensors, covered whole | 71 | 1.75e-5 | 9.94e-4 | fails at step 671 | 52 | 5.77e-5 | 4.36e-3 | |
| sensors, written | 72 | 3.90e-5 | 1.05e-3 | 0.080 | 43 | 9.26e-5 | 1.12e-2 | −1.72e-5 |
| a third of the fine run's steps | 71 | | | | 35 | | | |

Between the forms the tiled runs of the four problems differ by a few percent
or less. The last column is the composite mass of test/validation.jl's
cylindrical implosion on a level over 43 root nodes, relative to the exact
mass.

With the mask the level's own rate bounds nearly every root step, and no
bounding rate is mostly artificial diffusivity. Under the box of the 3-D blob
the root's nodes on the level's faces and beside them bound most steps
instead, at their hyperbolic rate.

| run | root steps / root-only | level steps / fine | class bounding the root step | overwritten nodes' largest CFL over the solver's |
|---|---|---|---|---|
| sod box | 0.941 | 0.996 | level 100% | 0.73 |
| sod tiles | 0.941 | 0.996 | level 100% | 0.73 |
| blobs box | 0.791 | 1.01 | level 79%, root covered 19%, root partly covered 1% | 0.98 |
| blobs tiles | 0.791 | 1.01 | level 99%, root covered 1% | 0.99 |
| shock box | 0.999 | 0.997 | level 99.4% | 1.00 |
| shock tiles | 0.996 | 0.994 | level 100% | 1.00 |
| blob3d box | 0.935 | 1.24 | root partly covered 49%, level 35%, root covered 16% | 0.93 |
| blob3d tiles | 0.761 | 1.01 | level 100% | 0.99 |

Before the mask, `max_rate` held a parent's overwritten nodes that lie at
least `LEVEL_BUFFER` nodes from any node a child does not fully cover to a
CFL number of 0.75 where the solver's was lower, which relaxed the bound the
covered coefficients set. The table's last column was taken under that
ceiling: with the mask those nodes stay at or below the solver's CFL number
in every run, so the ceiling no longer acted and was removed. Without it the
four problems reproduce the step counts and errors above to the digits
printed, as does an r-z capsule band run under a flat drive.

**Where the tiled overhead goes.** On tiles the level-wide artificial
coefficients and the ghost-flux divergence take about three fifths of the
step. Per level point the tiles cost within a factor of two of the box in
every phase but Sod's ghost-flux divergence, whose tiles have two interface
dimensions where the box spanning the channel has one. The same-level
synchronization exists only on tiles and is a few percent of the step. The
table gives milliseconds per root step of the box and the tiles, from the
sampling profile scaled by the timed wall, and the tiles' cost per level
point over the box's.

| phase | sod box | sod tiles | per point | blobs box | blobs tiles | per point | blob3d box | blob3d tiles | per point |
|---|---|---|---|---|---|---|---|---|---|
| level RHS and artificial coefficients | 6.71 | 10.0 | 1.23 | 70.8 | 92.2 | 1.13 | 407 | 1380 | 0.98 |
| Hermite shell fill | 0.53 | 1.24 | 1.93 | 4.61 | 4.87 | 0.92 | 127 | 255 | 0.58 |
| ghost-flux divergence | 3.09 | 9.11 | 2.42 | 28.3 | 33.1 | 1.02 | 168 | 608 | 1.05 |
| post-step shell | 0.06 | 0.10 | 1.35 | 0.45 | 0.51 | 1.00 | 11.9 | 23.5 | 0.58 |
| same-level sync | 0.01 | 0.64 | | 0.04 | 1.68 | | 0.2 | 79.5 | |
| level filter | 0.51 | 0.82 | 1.34 | 3.42 | 3.61 | 0.92 | 18.4 | 60.7 | 0.96 |
| level stage update | 0.41 | 0.82 | 1.64 | 1.71 | 2.56 | 1.30 | 8.0 | 37.9 | 1.38 |
| root RHS and Hermite endpoint | 2.53 | 2.70 | | 5.17 | 5.52 | | 31.9 | 30.5 | |
| regrid | 0.31 | 0.15 | | 5.19 | 3.85 | | 17.2 | 24.9 | |
| whole step | 15.3 | 26.6 | | 122 | 151 | | 801 | 2530 | |
| level points | 5967 | 7260 | | 27528 | 31623 | | 85778 | 295139 | |
| tiles on average | | 11.6 | | | 13.2 | | | 18.9 | |

| share of the tiles' step | sod | blobs | blob3d |
|---|---|---|---|
| level-wide artificial coefficients | 28.5% | 37.0% | 34.5% |
| ghost-flux divergence | 34.3% | 22.0% | 24.1% |
| Hermite shell fill | 4.7% | 3.2% | 10.1% |

The shell is filled only on a tile with a parent-fed face, and there only
over the boxes of the interpolation chain its slots read. Each tile's
right-hand side reads the primitives and velocity gradients computed in the
level-wide pass.

The 1-D shock's step divides in the same way, the folded box at the axis
taking a small share:

| shock, share of the step | box | tiles (7.1 on average) |
|---|---|---|
| level right-hand side | 34.2% | 16.9% |
| level-wide artificial coefficients | | 33.4% |
| ghost-flux divergence | 28.4% | 22.8% |
| root right-hand side and Hermite endpoint | 16.2% | |
| regridding | 3.1% | |
| shell fill | 1.7% | 3.0% |
| folded box at the axis | 1.9% | |

**Per level point and right-hand side.** The level-wide pass costs about an
eighth more per point than the fine grid's primitives, velocity gradients and
`compute_artificial!`. The ghost-flux divergence costs about three times the
fine grid's on Sod and two and a half times on the blobs, since it takes two
line solves where the fine grid takes one, plus the split of every flux into
its ghost-differenced part and the remainder. The divergence plans' one-sided
interface rows carry left-hand-side rows of their own, so the two solves
share no factorization. A scratch script (not kept) timed one evaluation at
the state after 150 root steps on the Sod and blob tiles, and after six on the
3-D blob's, against the uniform fine grid; each solve is summed over the
components and the interface dimensions.

| ns per point | Sod tiles | blob tiles | 3-D blob tiles |
|---|---|---|---|
| tiles, level points, padded block over its interior | 10, 6250, 1.74 | 13, 31213, 1.35 | 8, 125000 |
| level-wide pass | 102 | 114 | |
| of it, primitives and velocity gradients | 35 | 32 | |
| of it, exchanges between its stages | 4 | 2 | |
| of it, sensors and coefficients | 62 | 85 | |
| fine grid: primitives, velocity gradients and `compute_artificial!` | 90 | 102 | |
| fine grid: primitives and velocity gradients | 25 | 27 | |
| fine grid: `compute_artificial!` | 65 | 75 | |
| ghost-flux divergence | 71 | 71 | |
| of it, each of two line solves | 29 | 29 | 47 |
| of it, the split | 10 | 12 | 24 |
| fine grid's divergence, one solve | 23 | 29 | |

**The species band on the blobs.** The box carries the mass fraction further
below zero than the root alone: briefly at the regrid checks when the edge
spans most of a root spacing, and steadily, with nearly four times the
root-only run's steps, when it spans a third. At an edge of one root spacing
the box stays above the root-only minimum of the first configuration, and the
tiles stay above it even at a third of a spacing. A per-step trace of the
mass-fraction minimum to t = 0.4 (a scratch script, not kept):

| root nodes | edge, in root spacings | run | mass-fraction minimum | root steps |
|---|---|---|---|---|
| 64 | 0.64 | box, level 1 at the first two regrid checks, recovering within three steps | −0.064, −0.072 | |
| 64 | 0.64 | root only | −0.027 | |
| 32 | a third | box, covered root nodes, falling steadily | −0.26 | 339 |
| 32 | a third | root only | | 92 |
| | one | box | above −0.021 | |
| | a third | tiles | above −0.015 | |

### bench/archive/movinglevel.jl: disturbances a moving level carries

`julia --project=. -t 1 bench/archive/movinglevel.jl`, one rank and one thread on the
i9-12900K. The runs use Julia 1.11.4.

A fixed cover carries the uniform fine grid's disturbance to within a factor
of two under every filter pass and either derivative. A moving cover carries
far more, and the two masks of the parent's covered nodes, of its filter
residual and of its derivatives, together cut it by more than an order of
magnitude.

| moving cover, disturbance over the uniform fine grid's | neither mask | both masks |
|---|---|---|
| bubble, level, away | 260 | 8 |
| shock, beyond 0.05 | 9.4e4 | 140 |
| shock, beyond 0.02 | | 12 |

The configurations are two reduced tutorials:

| configuration | root | refinement |
|---|---|---|
| a helium bubble in air, edge width w = 10 mm (one root spacing) | 48² nodes 10 mm apart | tiles of edge 12 following the bubble 60 mm along the diagonal |
| the Imploding shock tutorial | a 256-node r-z line | one subcycled sensor-placed box |

For the bubble the disturbance is the largest difference of the mole fraction
from the exact solution more than 4w beyond its radius, over composite nodes
at the end. For the shock it is the largest departure of ρ from the gas at
rest ahead of the shock over every step, at nodes more than 0.02 and more than
0.05 inside the shock radius (the table's two columns). The shock radius lies
between 0.02 and 0.3 over the measurement. `-plain` filters the parent's
covered nodes as well (`MASK_CHILD_RESIDUAL[] = false`). The refined runs take
the parent's derivative mask, the package default
(`MASK_CHILD_DERIVATIVE[]`), and `-plainD` the plain derivatives
(`derivative_mask=both`). The shock runs accept the state at the axis near
t_c, which can carry a negative internal energy at a few nodes
(`validity = :permissive`). Single runs under the overwritten-node CFL ceiling
and the conservative coarse-fine coupling with its gate testing either side of
the face (2.4 minutes for both problems); the masses are in the conserved
quadrature.

| bubble | away, root | away, level | He mass | layout changes |
|---|---|---|---|---|
| coarse | 5.86e-3 | | 4.9e-15 | |
| fine | 3.97e-6 | | 4.4e-14 | |
| moving | 2.13e-6 | 3.09e-5 | −5.4e-6 | 2 |
| moving-plainD | 2.17e-6 | 3.68e-5 | +7.7e-7 | 2 |
| moving-plain | 3.11e-4 | 9.05e-4 | +1.0e-4 | 12 |
| moving-plain-plainD | 2.59e-4 | 1.05e-3 | +8.0e-5 | 13 |
| fixed | 1.34e-7 | 3.93e-6 | −1.3e-6 | 0 |
| fixed-plainD | 3.95e-8 | 3.93e-6 | −1.3e-6 | 0 |
| fixed-plain | 2.94e-6 | 3.93e-6 | −1.4e-6 | 0 |

| shock | root steps | beyond 0.02 | beyond 0.05 | layout changes | inadmissible |
|---|---|---|---|---|---|
| coarse | 673 | 7.50e-3 | 7.23e-4 | 0 | 0 |
| fine | 2023 | 2.91e-4 | 2.69e-7 | 0 | 0 |
| moving | 673 | 3.56e-3 | 3.83e-5 | 125 | 2 |
| moving-plainD | 762 | 1.92e-2 | 2.64e-4 | 105 | 0 |
| moving-plain | 675 | 2.60e-2 | 2.48e-2 | 152 | 3 |
| moving-plain-plainD | 762 | 4.37e-2 | 2.54e-2 | 171 | 0 |
| fixed | 673 | 2.96e-4 | 1.98e-7 | 0 | 0 |
| fixed-plainD | 797 | 2.22e-4 | 1.71e-7 | 0 | 2 |
| fixed-plain-plainD | 798 | 3.35e-4 | 2.44e-7 | 0 | 2 |

The shock's columns take the larger of the root and the level; the last
column counts the inadmissible nodes at t = 0.221, after the collapse.

The coarse-fine correction under the gate testing either side of the face
raises the moving shock's disturbance. With the correction off, or under the
default gate, which tests the uncovered nodes alone, the moving row comes
within a fifth of the fine grid's. How the correction does it is under
[bench/archive/reflux.jl](#bencharchiverefluxjl-the-conservative-coarse-fine-coupling).

| moving shock, beyond 0.02 | moving | moving-plainD |
|---|---|---|
| correction off (`conservative = false`) | 3.38e-4, 1.16 times the fine grid's | 1.08e-2 |
| default gate, testing the uncovered nodes alone | 3.05e-4, 1.05 times | |

The plain bubble run's mass change lies in the steps that changed the layout,
which account for +1.3e-4. With the derivative mask the shock's root takes 12
to 16% fewer steps.

**Attribution.** The root's filter pass is the cause of the moving bubble's
disturbance. Its residual at the covered nodes, where the restricted bubble
edge is a jump on the root spacing, spreads along the root's lines into the
uncovered nodes and into the covered ones the tiles' ghost layers are
interpolated from. Neither the regrid fill nor the level's own pass matters at
this level. A scratch script (not kept) replaced the state at regrids, reset
parent nodes to the exact solution after each step, or switched the filter
pass per level, on the plain bubble run (level, away: 1.19e-3). This and the
next table were taken at `d5fd609`, before the overwritten-node ceiling,
which leaves the bubble rows above unchanged:

| change | level, away |
|---|---|
| exact solution in every new tile at each regrid | 9.6e-4 |
| uncovered root nodes reset to the exact solution after each step | 1.7e-3 |
| every root node beyond R + 5w reset after each step | 4.6e-4 |
| no filter pass on the refined level | 1.2e-3 |
| restriction before the root's pass as well as after | 1.0e-3 |
| no filter pass on the root | 5.3e-5 |
| no filter anywhere (uniform fine grid without it: 3.6e-5) | 6.4e-5 |
| root pass without the residual of every fully covered node | 2.8e-5 |
| the same, the masked set eroded by 4 nodes | 9.0e-4 |
| every fully covered node masked, root reset beyond R + 5w after each step | 3.9e-6 |

**The resolution test.** Dropping the residual at every fully covered node
moves the smooth level rows of `test/convergence.jl`, the pass being weaker on
the covered side of the face than on the other:

| `test/convergence.jl` row | residual kept | every fully covered node masked |
|---|---|---|
| cascade-filter interface row, error at N = 48 | 1.43e-6 | 1.59e-6 |
| r-z axis row, errors at N = 96 and 192 | | 2.7 times those with the residual |

The mask therefore takes a covered node only where the relative undivided
fourth difference of the density along the line exceeds a threshold within a
reach of nodes. The default threshold of 0.01 within two nodes leaves the
convergence rows unchanged and keeps most of the bubble's gain from masking
every covered node:

| threshold, reach | bubble level, away | shock beyond 0.02 / 0.05, level | convergence level rows |
|---|---|---|---|
| 0 (every covered node) | 2.8e-5 | 1.5e-2 / 6.7e-4 | move |
| 0.001, 0 | 6.8e-5 | 1.9e-2 / 1.4e-3 | |
| 0.01, 0 | 1.55e-4 | 1.1e-2 / 2.8e-4 | unchanged |
| 0.02, 0 | 1.74e-4 | 1.3e-2 / 2.7e-4 | |
| 0.003, 2 | 4.05e-5 | 1.7e-2 / 1.4e-4 | move |
| 0.01, 2 (default) | 3.91e-5 | 4.3e-2 / 1.1e-4 | unchanged |

"Unchanged" is every printed digit of every row from the interface window to
the temporal order section; under the overwritten-node ceiling the default
leaves them unchanged as well. This table predates the derivative mask. The
first of the shock's two figures varies about fourfold over these settings
without a trend. It is the precursor the parent's derivatives place ahead of
the shock, which the next paragraphs attribute.

**The parent's derivatives.** The precursor ahead of a moving shock's level
is the alternating tail of the root's compact derivative across the
restricted shock, which reaches the face node from the covered nodes behind
it. Masking the root's divergences and gradients at the covered nodes
together removes it, and either alone does not. Under the plain derivatives
the precursor falls as `tag_buffer` widens, every run with the positivity
limiter on (`limiter=true`) ending admissible:

| `tag_buffer` | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| shock's level beyond 0.02, plain derivatives, limiter on (fine grid: 1.93e-4) | 9.33e-3 | 6.58e-3 | 5.25e-3 | 2.51e-3 | 1.22e-3 |

The fill of the nodes a regrid newly covers changes the level's disturbance,
but the fill only carries it inward, since a fixed box with its inner face
ahead of the shock (`fixed_inner=0.1`, the face at r = 0.0881) carries it as
well.

| `tag_buffer` 8, nodes a regrid newly covers | level, beyond 0.02 |
|---|---|
| set to the state at rest | 3.73e-4 |
| filled from the uncovered root alone | 2.54e-3 |

On that fixed box a scratch script (not kept) switched one candidate off at a
time, the level's maximum falling at the face node when the shock is six root
spacings inside it:

| change, fixed box at r > 0.0881, limiter on | level, beyond 0.02 |
|---|---|
| none | 4.61e-3 |
| no filter pass on the root | 6.46e-3 |
| no filter pass on the level | 6.01e-3 |
| root pass without the residual of every covered node | 4.68e-3 |
| uncovered root nodes ahead of the shock reset to rest after each step | 4.48e-3 |
| artificial properties off | 1.01e-2 |
| root right-hand side without the covered state 5–20 nodes inside the face | 1.66e-4 |
| the same, nodes 5–12 | 1.02e-3 |
| the same, nodes 5–8 | 1.40e-2 |
| the same, for the artificial coefficients only | 6.31e-3 |
| the same, for everything but the artificial coefficients | 1.46e-4 |
| root update zeroed at the restricted nodes | 1.10e-2 |

The disturbance follows the distance from the face of the nearest strong
jump in the root's right-hand side, and each hidden range leaves a jump of
its own at its far end. At one state, with the shock seven spacings inside the
face, the energy rate at the face node and the four covered nodes behind it
alternates in sign and grows toward the shock with the restricted shock in
view, and stays below 1 without it: the alternating tail of the compact
derivative, decaying by 0.4 per node toward the face.

| energy rate, shock seven spacings inside the face | face node | first covered | second | third | fourth |
|---|---|---|---|---|---|
| restricted shock in view | −12.8 | 34 | −86 | 215 | −540 |

| limiter off, masked at the covered nodes | level, beyond 0.02 |
|---|---|
| nothing (reference) | 2.09e-3 |
| divergences alone | 2.20e-3 |
| gradients alone | 4.10e-3 |
| both | 1.21e-4 |

The mask's variants were chosen with the limiter on, its registers holding the
masked divergence unless a row says otherwise, on the fixed box and the moving
box at `tag_buffer` 4:

| variant | fixed | moving |
|---|---|---|
| plain derivatives | 4.61e-3 | 9.33e-3 |
| mask from the fourth node inside a face (as delivered) | 8.05e-5 | 3.25e-4 |
| the same, the dropped source applied at the masked nodes | 8.78e-4 | 2.81e-3 |
| mask from the overwritten nodes, the fifth inside | 5.73e-4 | 1.33e-2 |
| registers holding the plain divergence | 2.59e-3 | 7.12e-3 |
| registers restoring the dropped source at the masked nodes | 8.05e-5 | 3.29e-4 |

The last arrangement gives 2.61e-4 at `tag_buffer` 8, every run ending
admissible. Under the limiter the mask nevertheless stays off, even when it
is enabled, because neither register arrangement holds up on the limiter's
refined rows of `bench/archive/positivity.jl part=levels`. Registers holding the
masked divergence carried its dropped flux to every face beyond the masked
nodes, and registers restoring the dropped source left inadmissible nodes and
unguaranteed faces where the plain derivatives leave none.

| limiter's refined rows | cylindrical Noh box (`noh43`) | Woodward–Colella |
|---|---|---|
| plain derivatives | plateau 14.83, shock radius 0.2086 | no unguaranteed faces |
| registers holding the masked divergence | plateau 12.86, shock radius 0.2282, 19.8% of the stage faces limited | lost positivity at step 1140 |
| registers restoring the dropped source at the masked nodes | 31 shell and 42 other inadmissible nodes | 934 unguaranteed faces |

**Composite conservation.** Uncorrected, the derivative mask loses far more
mass than the plain derivatives, since the removed tail carries part of the
mass the parent's uncovered nodes exchange through the coarse-fine face. The
correction returns nearly all of it. The table gives the relative mass drift
at t = 0.2 in the conserved quadrature of the Sod shocks of
`test/level_tests.jl`, which cross a level in both directions, with the
coarse-fine correction and without it (the uncorrected columns without Ω,
since that coupling builds no captures). The table was taken at `d346d35`,
with the correction's gate testing either side of the face; under the default
gate, which tests the uncovered nodes alone, the masked crossings drift by the
figure in the default row of the gate table of
[bench/archive/reflux.jl](#bencharchiverefluxjl-the-conservative-coarse-fine-coupling):

| case | plain | mask | plain, uncorrected | mask, uncorrected |
|---|---|---|---|---|
| global step, C6 | +6.2e-8 | −8.8e-7 | +8.5e-5 | −5.2e-4 |
| global step, `:d8` and pentadiagonal filter | −5.4e-9 | −1.4e-7 | +2.6e-6 | −5.3e-4 |
| subcycled, C6 | +2.3e-8 | +3.7e-8 | +9.1e-5 | −7.5e-4 |
| subcycled, C10 | +3.8e-9 | −7.4e-9 | +7.0e-5 | −2.0e-3 |

**The tutorial configuration.** On the Advected bubbles tutorial itself the
masked run's disturbance away from each bubble lies within a factor of ten of
the uniform fine grid's, where the plain pass leaves one to four hundred times
it, and each gas's mass change is one to two orders of magnitude smaller. The
disturbance is the largest difference from the exact solution more than 4w
beyond each bubble's radius, helium / SF6 / krypton.

| run (96² root, three bubbles, tiles of 12) | away from each bubble | mass change, air / He / SF6 / Kr |
|---|---|---|
| refined | 3.9e-5 / 4.9e-6 / 1.3e-5 | 1.2e-8 / −1.1e-6 / 7.4e-7 / −1.6e-6 |
| refined, plain pass | 1.6e-3 / 4.7e-4 / 7.8e-4 | 7.3e-7 / −5.5e-5 / −9.7e-6 / −6.1e-5 |
| uniform 288² | 4.0e-6 / 3.9e-6 / 3.9e-6 | round-off |

**Regrid churn.** The mask of the covered nodes' residual quiets the tag
history: tiles beside a survivor are created later, and the leading tile is
no longer dropped and re-created. The MPI suite's deep regrid runs the masked
pass, under which a tile of its second level still changes owner.

| case | plain pass | mask |
|---|---|---|
| tiled Sod fixture of the level tests (201 nodes, tiles of 8, checks every 5 steps): first check to create a tile beside a survivor | after step 8 | after step 65 |
| tiled run whose first tiles a Dirichlet inflow's shock creates | the leading tile dropped and re-created four times in 160 steps (at `d5fd609`) | a tile added at the checks after steps 20, 40, 95, 140 and 185, none dropped |
| MPI suite's deep regrid (121-node Sod tube, diaphragm at x = 0.35, tiles of 4, three levels, np = 8): checks within 240 steps at which a second-level tile changes owner | 5 | 2 |
| refined Advected bubbles tutorial, single runs: wall, tiles at the end | 40 s, 25 | 35 s, 23 |

### bench/archive/levelpattern.jl: the grid-scale pattern of a refined level

`julia --project=. -t 8 bench/archive/levelpattern.jl` and `julia --project=. -t 8
bench/archive/levelpattern.jl problems=interface configs=uniform,box,tiles`, one rank
and eight threads on the i9-12900K, single runs. The runs use Julia 1.11.4 and
the package at the commit that adds the script unless a row says otherwise.
The pattern is the mean undivided fourth difference of the density over the
shell behind a converging shock, relative to the density, or of the helium
mass fraction over a sharpened interface; the script's header defines both
and the problems.

**The converging shock.** A box reproduces the uniform run's pattern, and
tiles, fixed or regridded, carry about a fifth more. Without the mask of the
covered nodes' residual the regridded tiles' pattern doubles while the fixed
tiles' moves in the fourth digit only, so the mask removes what regridding
added and leaves fixed and regridded tiles with the same remainder. The plain
pass filters the parent's covered nodes with their residual (`mask=false`).
The last column counts the nodes the state report finds inadmissible at the
end.

| configuration (64² root, t = 0.165, R = 0.167, 740 shell nodes) | root steps | pattern | ratio | inadmissible |
|---|---|---|---|---|
| uniform 191² | 629 | 1.119e-2 | 1.00 | 5 |
| box | 628 | 1.118e-2 | 1.00 | 5 |
| tiles of 8 | 628 | 1.331e-2 | 1.19 | 12 |
| regridded tiles of 8 | 630 | 1.335e-2 | 1.19 | 12 |
| tiles, plain pass | 628 | 1.330e-2 | 1.19 | 12 |
| regridded tiles, plain pass | 646 | 2.634e-2 | 2.35 | 13 |

The same measure, with snapshots every 0.015 of simulated time, gave the
regridded tiles 2.39 at `ce1f513`, before the overwritten-node ceiling
(`f1d420c`) and the mask of the covered nodes' residual (`04d3cff`). Over
the run (`dsnap=0.015`) the regridded tiles' excess over the uniform run is
largest early, vanishes in mid-run and returns toward the end:

| t | 0.03 | 0.045 | 0.06 to 0.12 | last three snapshots |
|---|---|---|---|---|
| regridded tiles' pattern over the uniform run's | 1.38 | 1.13 | 1.00 to 1.04 | 1.07, 1.16, 1.19 |

**The sharpened interface.** Smoothed over the level
(`_level_sharpening_gradients!`), the sharpening flux's filtered gradients
leave the tiles a hundredth of the difference from the uniform run that
smoothing tile by tile left, and twice the difference the tiles carry without
the flux. The table gives the mean and largest difference of the helium mass
fraction from the uniform run over the band:

| configuration (48² root, w = 10 mm, tiles of 8, `C_sharpen = 1` and 980 steps unless the row says otherwise, 500 band nodes; uniform 144² pattern 3.066e-2) | ratio | mean | largest |
|---|---|---|---|
| box | 1.00 | 1.03e-7 | 9.23e-7 |
| tiles, gradients smoothed per tile (`7d6b0ba`) | 0.98 | 1.79e-2 | 4.60e-2 |
| tiles, gradients smoothed over the level | 1.00 | 1.75e-4 | 1.19e-3 |
| tiles, `C_sharpen = 0` (245 steps) | 1.00 | 8.68e-5 | 5.03e-4 |

Smoothed tile by tile, the flux's filtered gradients took at a shared face
the smoother's rows for a field without ghosts, and the gate and the pair
normals changed along the tile lattice. The level test `tiled level: sharpening
gradients across shared faces` checks the smoothed gradients against one
patch over the same region.

### bench/archive/amr_balance.jl: rebalance and migration mechanics

`mpiexec -n 4 julia --project=. -t 1 bench/archive/amr_balance.jl [N=800] [interval=10]
[steps=200] [persist=1]` (`603af33`).

Settled: mechanics only, and the reason `rebalance` and `rebalance_persist`
exist. Without rebalancing the stored groups drift out of balance; with it on
at its most eager settings every check repartitions, and the partition then
follows the timing noise from check to check. The workstation cannot say what
a rebalance is worth, since per-rank costs on rzhound and rzadams differ from
it by 27–66× and move with rank placement. A moved tile here is one kilobyte.
The case is a 1-D Sod shock crossing a tiled refined level at np = 4.

| rebalancing | tiles per rank | per-check max/mean busy time |
|---|---|---|
| off | two on ranks 0–1 against four on 2–3 | climbs to 1.3–1.5 |
| on, threshold 1 and persist 1 | one per rank where the count allows, repartitioned at every check | 1.0–1.7 |

`MIGRATION_AUDIT` holds the migrated state bitwise against the replicated
carry it replaced, at zero differing slots, in the MPI suite.

### bench/archive/amr_scaling.jl: weak scaling of the level coupling

`mpiexec -n <np> julia --project=. -t 1 bench/archive/amr_scaling.jl [N=96] [tile=8]
[dim=2] [steps=3] [reps=20]`; before is `235d557`, after is `d54f019`.

The distributed level coupling weak-scales as the root does: its memory
follows the rank's own tiles, and the refined step grows no faster than the
unrefined root's. The case is a planar slab of tiles spanning a root of
64 × N·np nodes under global stepping, so each rank holds about the same root
block and tile count at every np. Each figure is the largest over ranks: the
level's memory (`Base.summarysize` of the `Level`, which excludes the tiles'
patches), the median of twenty `sync_levels!` calls, and one step, beside one
step of the same root unrefined.

| np | tiles | level memory, before / after | `sync_levels!`, before / after | step, before / after | root alone |
|---|---|---|---|---|---|
| 1 | 20 | 6.23 / 4.73 MB | 1.25 / 1.08 ms | 30.8 / 33.5 ms | 4.9 ms |
| 2 | 44 | 13.14 / 5.34 MB | 5.20 / 1.26 ms | 64.5 / 42.7 ms | 5.8 ms |
| 4 | 92 | 26.79 / 5.68 MB | 7.25 / 1.40 ms | 151.9 / 54.0 ms | 11.5 ms |
| 8 | 188 | 54.22 / 6.21 MB | 23.00 / 2.84 ms | 360.9 / 80.3 ms | 15.3 ms |

Before, every rank of the parent held every tile's chain, box and Hermite
storage and gathered every tile's box with one Allgatherv per tile per stage,
so the level memory was 0.29 MB per tile of the whole level on every rank and
the step grew 12× from one rank to eight. After, the remaining growth of the
level memory is the four more tiles and the per-tile metadata every parent
rank keeps. The refined step grows 2.4× from one rank to eight where the
unrefined root grows 3.1×. The root's growth is the workstation's own
weak-scaling loss, from the hybrid cores and the planar pathology at eight
ranks above. Single runs; read the growth, not the third digit. The two
couplings agree bitwise: three-dimensional tiled runs at four and eight
ranks, subcycled and global-step, regridding and static, viscous with ghost
fluxes and with tiles over one rank and over four, reproduce every patch
interior bit for bit after five steps.

### bench/device_solver.jl: the device battery

`julia --project=<env-with-AMDGPU> -t 8 bench/device_solver.jl backend=amdgpu`.
RX 6800 XT, AMDGPU.jl on Windows/HIP; residency and stacked storage `602002c`
and `346f83e`. Workstation numbers are evidence that the structural pitfalls
are gone, not a performance claim about the target machine. RDNA2 runs vector
double precision at a sixteenth of the single-precision rate, where the
target machine runs it at full rate.

Settled: the CPU keeps `@threaded`; the one unconditional fence is the
reduced-solve one; the device floor is launch submission, not arithmetic.

| measurement | value |
|---|---|
| 64³ TGV full step, device F64 / F32 | 0.146 / 0.117 s per step |
| the same on the 8-thread CPU | ~0.12 / ~0.10 s per step |
| isolated flux assembly, 64³ two-species | 9.9× the CPU |
| staged halo and pair copies | 0.6–6.6% of device wall |
| reduced-interface copies | 2–5% of device wall |
| first-launch kernel compilation | ~9 s per body |
| KA-CPU against `@threaded` at 64³ | 2.8× (flux assembly) to 40–50× (RK update) slower |
| removing the per-launch synchronize | 28–32% off the device step |

**Stacked tile storage.** Stacking a level's tiles into one array puts the
twelve-tile 3-D level ahead of the CPU, the first tiled configuration on which
the device leads. Warm steps, one run each, the 8-thread CPU as reference:

| case | before stacking | after |
|---|---|---|
| 1-D tiled regridding Sod, 8 tiles of 25 fine nodes, subcycled | 0.11 s (CPU 1.1 ms) | 0.049 s |
| two-slab viscous wave | 9 ms (CPU 0.1 ms) | 9.8 ms (root level is not stacked) |
| 3-D level, twelve 16³ tiles in one stack, subcycled | — | 0.354 s (CPU 0.576 s) |
| unstacked 64³ TGV | 0.146 s | 0.164–0.176 s |

The residual floor is the work that stays per tile: the shell impositions,
the interface records, and `max_rate`'s two reductions per tile. Whether the
fill and scatter kernels' fourth index dimension of extent one costs anything
on the unstacked TGV is at the edge of the run-to-run spread and needs a
repeated-process measurement.

**Precision.** Single precision is not the default because of its
mean-density drift. On the device it runs only a quarter faster, because the
step is bounded by launch submission and fences rather than arithmetic.

| Float32 against Float64 | |
|---|---|
| CPU, 64³ TGV to t = 10: peak dissipation | identical to the printed precision |
| the same: footprint | 2.00× smaller |
| the same: wall | 1.10× |
| the same: mean-density drift | 1.4e-4 against 7.5e-13 |
| device: rate | 1.25× the Float64 rate |

### bench/device_mpi.jl: distributed device runs

`mpiexec -n 2 julia --project=<env-with-AMDGPU> bench/device_mpi.jl
backend=amdgpu` (repeat at 4 and 8; `6f14a0c`). Every distributed device stage
was accepted on `max |device − cpu| = 0` over full runs at two, four and eight
ranks, in both launch-policy modes. The staged transfer volumes are the halo and
reduced-interface percentages recorded above.

### bench/archive/device_nasa9.jl: NASA-9 on the device

`julia --project=<env-with-AMDGPU> -t 8 bench/archive/device_nasa9.jl backend=amdgpu`,
RX 6800 XT, He/CO2/N2 unless noted. The recovery set is hard states (joins
to a few ulp, range edges and beyond, no-root energies, no-bracket
compositions) and a temperature sweep, sized in the table's first row. On the
KA CPU backend the mirror is bitwise against the host
(`test/device_tests.jl`); on the GPU the logarithm and the integer powers are
the device's own.

| measurement | Float64 | Float32 |
|---|---|---|
| recovery set | 120 hard states and 100,000 swept from 150 to 30,000 K | |
| recovered T_ion, max relative difference | 2.9e-14 (130 eps) | 1.7e-5 (142 eps) |
| cp, max relative difference | 5.7e-15 | 2.8e-6 |
| points whose status differs, of 100,120 | 0 | 24, all met/missed convergence |
| the same at 8 species (`:polynomial`, of 20,220) | 3 | 71 |
| wall case, 20 steps, max component difference / max component | 1.3e-13 | 1.2e-4 (Float64 against Float32 on the CPU: 6.0e-5) |
| He/CO2 shock tube, 200 steps | 6.1e-15 | 3.7e-6 (Float64 against Float32: 1.5e-5) |
| refined wave through the ghost flux, 12 steps | 2.0e-14 | — |
| 8-species wall case, 10 steps | 3.2e-14 | — |
| mirror size that still launches | 4.2 kB (16 species, three intervals) | |
| 48³ recovery, CPU 8 threads / device, NASA-9 | 2.96 / 1.05 ms | 3.64 / 0.22 ms |
| the same, `IdealMixture` | 0.52 / 0.35 ms | 0.23 / 0.19 ms |
| 48³ step, CPU / device, NASA-9 | 146 / 185 ms | 127 / 144 ms |
| the same, `IdealMixture` | 110 / 174 ms | 98 / 149 ms |

Settled: the device reproduces the host recovery to the inversion's own
tolerance. At three species a status differs only where one side meets the
32 eps criterion and the other misses it; at eight species under
`:polynomial` three points differ in another bit, in a sweep where the
polynomial extension leaves 6286 host points unrecovered. No pitfall: a
`Nasa9Mixture` step costs 6% more than an `IdealMixture` step on the device,
against 33% on the CPU. The single-precision misses on the host itself sit at
the 6000 K join. The others lie near 10–11 kK.

### probes/device_floors.jl: rzadams floors and the wait stall

`julia --project=<env> probes/device_floors.jl` on the MI300A APUs of
rzadams, logs in `bench/archive/logs/rzadams_20260819*.txt`. The ROCm version is
6.4.3.

| measurement | value |
|---|---|
| kernel submission | 10 µs |
| launch + synchronize round trip | 25 µs |
| line solves | 0.14–0.40 ms per apply |
| 64³ TGV over 4 APUs | 0.074–0.088 s per step |
| F64 / F32 whole-step ratio | ~1.2 |
| 256³ single-species TGV over 4 APUs | 0.35 s per step baseline; 24,490 steps to t = 10 in 3.69 h |

Open: an intermittent stall mode in which every device wait costs an integer
number of milliseconds (13.000 ms medians) for seconds to beyond 30 s. It sits
below the Julia layer and inflated the 256³ run's solver average to 0.52 s per
step even at `-t 1`. `reference/bugreports/rocm_wait_stall_report.md` has the
characterization. Until it is resolved, run device-resident rzadams jobs at
`-t 1` per rank; a wall number from a multithreaded process is untrustworthy
without a stall watch beside it.

### Dead ends and one-line lessons

- An explicit Gaussian pass ahead of restriction (Pyranda's `c4ff3` role):
  cuts the shock undershoot 2.8× but raises the total round-trip error. Kept
  as a tool, not a default.
- A `::Type` argument through the `pointwise!` launcher: 9× as silent
  per-point runtime dispatch.
- Holding the `FieldVector` tuple form on the host: 3× on `assemble_fluxes!`.
- A keyed, `Any`-typed staging-buffer cache: 33 dispatch sites in
  `compute_rhs!`'s jetcheck report without ever executing. Reverted.
- A kernel-argument tuple longer than 32 elements: `InvalidIRError` on device.
- KA-CPU equality cannot certify the device path: four defects passed every
  KA-CPU test and failed only on real device storage.
- Judge a cost case on the quantity the refinement predicts, not on pointwise
  in-region error.
