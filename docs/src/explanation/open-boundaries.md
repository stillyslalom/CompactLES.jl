# Characteristic open boundaries

At a subsonic boundary, some information travels into the domain and some
travels out. Prescribing the complete primitive state replaces both sets and
reflects outgoing disturbances. Navier--Stokes characteristic boundary
conditions (NSCBC) instead alter only incoming wave amplitudes.

## Local wave picture

Ignoring transverse and viscous terms momentarily, the normal Euler equations
can be decomposed into acoustic, entropy, vorticity, and species waves. Their
normal characteristic speeds are `u_n-c`, `u_n`, and `u_n+c`. The face normal
and local Mach number determine which waves enter.

CompactLES uses a local-one-dimensional-inviscid approximation (LODI). The
one-sided compact derivative first evaluates the ordinary boundary RHS. NSCBC
then replaces selected incoming amplitudes and maps only their difference back
to conserved-variable time derivatives.

## Subsonic outflow

At a subsonic outflow, one acoustic wave enters. `NSCBCOutflowBC` replaces its
amplitude by a pressure-relaxation model,

```math
L^* = K(p-p_\infty),\qquad
K=\sigma(1-M^2)c/L_{ref}.
```

Here ``L^*`` is the imposed incoming acoustic wave amplitude, ``p_\infty`` the
far-field pressure target (`pinf`), ``M`` the local Mach number, ``c`` the sound
speed, ``\sigma`` the relaxation coefficient (`sigma`), and ``L_{ref}`` the
relaxation length (`Lref`), which sets the scale over which the relaxation acts
and defaults to the domain extent normal to the face.

This does not impose `p=pinf` pointwise. It slowly controls the mean pressure
while permitting outgoing waves to leave. `sigma` balances pressure anchoring
against reflection; its appropriate value depends on the domain length and
unsteadiness.

Supersonic outflow receives no correction because every inviscid characteristic
leaves the domain.

## Subsonic inflow

At a subsonic inflow, all but one acoustic wave enter. `NSCBCInflowBC` relaxes
the incoming acoustic, entropy, transverse-velocity, and species waves toward
target velocity, temperature, and composition while preserving the outgoing
acoustic wave computed by the interior solution.

Targets may be constant or supplied as a pointwise function of position and
stage time. The target must be thermodynamically meaningful for the selected
EOS.

## Transverse and viscous effects

The LODI approximation separates normal waves. `beta_t` controls how much of
the transverse contribution is carried by the imposed acoustic wave; a negative
value selects the local-Mach recommendation. Viscous boundary terms are retained
as computed and are not included in the characteristic derivation.

Current NSCBC is intended for faces whose normal coordinate has unit scale
factor: Cartesian faces, cylindrical radial or axial faces, and spherical
radial faces. Angular faces require additional metric terms in the wave
analysis and should not be assumed equivalent.

## EOS dependence

Wave speeds and primitive amplitudes are general, but mapping a pressure-wave
correction to total energy depends on

```math
\phi=\left.\frac{\partial(\rho e)}{\partial p}\right|_{\rho,Y}.
```

Composition relaxation additionally needs derivatives of ``\phi`` with respect
to mass fractions. These are part of the EOS extension contract, so the
ideal, NASA-9, and stiffened-gas models share one boundary implementation.

## Parallel ordering

NSCBC evaluates full-field compact derivatives before selecting the rank-local
boundary plane. Those derivatives are collective along a decomposed direction,
so every rank must enter them even if only one rank owns the face. A boundary
implementation that returns before the collective will deadlock.

The same requirement explains why a `SwitchableBC` must change on every rank at
one completed step. Use a callback trigger, never an unreduced local condition.

## Validation responsibility

Non-reflecting is an approximation, not a mathematical guarantee. Evaluate a
boundary by comparing the quantity of interest with a longer domain or a known
outgoing-wave solution. A small reflection coefficient for one frequency,
angle, and Mach number does not validate a boundary for all disturbances.
