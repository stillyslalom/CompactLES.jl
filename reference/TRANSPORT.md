# Transport physics plan

CompactLES currently has a single-temperature, mass-gradient Fickian species
flux.  This file defines what that model can represent, the evidence required
for hydrogen-isotope transport and other declared HED materials, and the physical
closures needed before it can represent high-energy-density plasma transport.
The software ownership, query, cache and analytic fast-path contracts are in
[DESIGN.md](DESIGN.md#material-and-physics-interfaces); this file owns physical
model assumptions, data coverage and transport verification.

## Present contract and its limit

At each point and in each direction, `src/rhs.jl` assembles

```
J_k = rho * (-D_k * grad(Y_k) + Y_k * sum_j(D_j * grad(Y_j))).
```

The correction velocity enforces `sum(J_k) = 0`.  `CeaTransport` can obtain the
mixture-averaged `D_k` from symmetric binary coefficients, whose present data
model scales every pair as `T^n/p`.  This is a useful dilute-neutral-gas model.
It has no driving terms for pressure, ion or electron temperature, or electric
potential, no interspecies friction matrix, no charge or ionization state, and
no zero-current or ambipolar closure.  The evolved state also has one material
temperature, named `T_ion`; `T_ele` does not yet exist.

The bundled CEA thermodynamic table currently has exact records for `H`, `H+`,
`D`, `D+`, `H2`, `D2`, `HD`, and `e-`, but not `T`, `T+`, `T2`, `HT`, or `DT`.
Its transport table has pure-species records only for `H`, `H2`, and `D2` from
that set.  Adding a tritium transport coefficient cannot make tritium an
evolvable species: sourced thermodynamics and the appropriate EOS/ionization
state are prerequisites.

Keep neutral atoms (`H`, `D`, `T`) distinct from neutral molecules (`H2`, `HD`,
`HT`, `D2`, `DT`, `T2`) and both distinct from ions (`H+`, `D+`, `T+`).  A
molecular measurement is not evidence for atomic transport.  An ion model is
not valid merely by inserting `Z = 0`, and a neutral-gas model is not valid by
inserting a charge.

## Dilute neutral hydrogen isotopes

The immediate model should store pair-specific fits to the binary molecular
diffusivity at a stated reference pressure,

```
log(D_ij(T, p_ref)) = sum_q a_ijq * log(T / T_ref)^q,
D_ij(T, p) = D_ij(T, p_ref) * p_ref / p.
```

Each pair needs species identities, `Tmin`, `Tmax`, units, source, phase and a
fit residual.  Evaluation outside the source range must be an explicit policy.
This representation preserves dilute-gas reciprocal-pressure scaling and the
temperature curvature that a common exponent loses.  It is not a dense-fluid
model.

Primary evidence, in preferred order:

1. Marrero and Mason's critical evaluation includes recommended `H2-D2`
   correlations and the underlying measurements: [*Gaseous Diffusion
   Coefficients*, J. Phys. Chem. Ref. Data 1, 3--118 (1972)](https://doi.org/10.1063/1.3253094),
   with a [public NIST scan](https://srd.nist.gov/jpcrdreprint/1.3253094.pdf).
   It covers neutral dilute
   gases only.  Correlation ranges and stated pair-specific uncertainty must be
   transcribed and independently checked against the plotted values.
2. Müller and Klemm measured `H2-HD`, `H2-D2`, and `HD-D2` at 24 C and 760
   Torr and jointly fit older measurements involving `HT`, `DT`, and `T2`:
   [*Diffusion in Binary Mixtures of H2, HD and D2 at 24 C*, Z. Naturforsch.
   25a, 243--246 (1970)](https://doi.org/10.1515/zna-1970-0216).  The new
   measurements report about one-percent error; the tritium-containing values
   are older and less precise.  Use the paper's actual tabulated values and
   labels, never a mass-scaled substitute.
3. Song and Wang calculate low-density diffusion and thermal-diffusion factors
   for hydrogen isotopologues with modern ab-initio interaction potentials over
   298--2000 K: [J. Chem. Eng. Data 61, 1910--1916 (2016)](https://doi.org/10.1021/acs.jced.6b00076).
   These are calculated values,
   compared with limited experiments, rather than new measurements.  The free
   supporting information is the useful numerical artifact; ACS copyright does
   not by itself grant redistribution, so record a reproducible extraction and
   seek permission before vendoring its tables.

The familiar reduced-mass factor is only a controlled limiting test.  For two
species sharing the same classical interaction potential, first-order
Chapman--Enskog theory gives a factor proportional to
`sqrt(1/m_i + 1/m_j) / Omega_ij(T)`.  Real isotopologues can have different
quantum and internal-state collision integrals.  Do not use this factor to
manufacture missing tritium data.  Atomic `H/D/T` needs a separate atomic
collision source or a documented potential/scattering calculation.

## Ionized H/D/T mixtures

A collisional two-ion plasma flux is driven by more than composition.  In a
binary notation its relative mass flux contains concentration diffusion plus
barodiffusion, ion thermodiffusion, electron thermodiffusion and
electrodiffusion.  The electric field is constrained by the electron momentum
model and, in the usual low-frequency quasineutral limit, by current closure;
it is not an arbitrary additional Fick coefficient.  Kagan and Tang derive the
electrodiffusion term and show its charge-to-mass dependence in
[*Electro-diffusion in a Plasma with Two Ion Species*](https://doi.org/10.1063/1.4742162),
[open manuscript](https://arxiv.org/abs/1204.1312).  Their companion analysis shows that plasma
thermodiffusion can be comparable to or larger than barodiffusion:
[*Thermo-diffusion in Inertially Confined Plasmas*](https://doi.org/10.1063/1.4851715),
[open manuscript](https://arxiv.org/abs/1310.8227).

For weakly coupled, classical ions, a screened-Coulomb Burgers/Chapman--Enskog
model supplies resistance and thermal-diffusion coefficients.  Paquette et al.
numerically evaluate the screened-Coulomb collision integrals and document the
dilute limit in [*Diffusion Coefficients for Stellar Plasmas*, ApJS 61,
177--195 (1986)](https://doi.org/10.1086/191111), [public scan](https://adsabs.harvard.edu/pdf/1986ApJS...61..177P).
This is a useful
independent weak-coupling reference, but its static screening and classical
assumptions must be checked against the intended HEDP state.

Across moderate coupling, Stanton and Murillo give fitted effective-Boltzmann
collision integrals and compare self-diffusion, interdiffusion, viscosity and
conductivity with molecular dynamics in [*Ionic Transport in High-Energy-
Density Matter*, Phys. Rev. E 93, 043203 (2016)](https://doi.org/10.1103/PhysRevE.93.043203).
Baalrud and Daligault instead use
the potential of mean force in [*Effective Potential Theory for Transport
Coefficients across Coupling Regimes*, Phys. Rev. Lett. 110, 235001 (2013)](https://doi.org/10.1103/PhysRevLett.110.235001),
[open manuscript](https://arxiv.org/abs/1303.3202).  Effective-potential theory extends binary
kinetics into strong coupling but breaks down as caging and potential-energy
transport dominate.  Neither source licenses a claim of universal warm-dense
matter accuracy; degeneracy, partial ionization, dynamic screening and the
model's stated coupling range are acceptance criteria.

For fully ionized `H+`, `D+`, and `T+`, the interaction potential is isotope
independent to the nonrelativistic Born--Oppenheimer accuracy normally used,
while masses enter the kinetic collision integrals explicitly.  That statement
supports evaluating one published ion model with the actual isotope masses. It
does not support rescaling a measured neutral coefficient or substituting an
`H-H` table for `D-T`.  Partial ionization requires charge-state populations
and neutral-ion collisions as separate inputs.

The paper's Appendix C fits and its interdiffusion equations are sufficient for
a bounded, coefficient-only implementation now.  The authors' live
[Stanton--Murillo evaluator](https://tempest-stc.msu.edu/tools/stanton-murillo-transport)
provides an independent numerical oracle; archive input/output cases with the
test provenance rather than depending on the web application at test time.
Implement the published equations from the paper and cite them.  No external
numerical table needs to be redistributed.  The paper and web tool do not state
a software license for copying their implementation, so do not copy page source
or scripts without permission.

The bounded H4a reference uses the paper's finite-temperature Thomas--Fermi
electron screening approximation, Eq. 25, while fixing every isotope at
`Z = 1`; the calculator's Thomas--Fermi checkbox changes mean ionization and is
therefore left off for the oracle cases.  H4a currently accepts only
`theta = k_B T/E_F >= 10`.  That conservative software-domain threshold keeps
the reference in a hot, weakly degenerate regime; it is not a physical phase
boundary, an ionization criterion, or a switch for the cold-to-warm model.

## Required interface

The following are requirements on the rich material/flux path in
[the interface design](DESIGN.md#material-and-physics-interfaces), not additions
to every scalar transport query. The ideal analytic path retains its direct
recovery and `(mu, kappa, D)` contract without population or plasma workspaces.

Separate coefficient evaluation from flux closure.  A future plasma transport
provider should consume the shared local thermodynamic and ionization state, including
`rho`, number fractions, isotope masses, charge states, `T_ion`, `T_ele`, and
the screening/degeneracy state required by the selected model.  It should
return symmetric interspecies resistance coefficients and the thermal-force
coefficients with their validity diagnostics.  A plasma species-flux closure
then solves the constrained multicomponent system using `grad(x)`, `grad(p_i)`
or the equivalent chemical-potential force, `grad(T_ion)`, `grad(T_ele)`, and
the electric-field/electron-momentum closure.  It must enforce zero total mass
diffusion and the selected current constraint to solver tolerance.

The scalar `transport_at(...).D` result cannot carry those coefficients.  The
flux routine must accept a transport result type and dispatch to neutral
mixture-averaged or plasma multicomponent assembly.  The diffusive timestep
estimate must use a justified spectral bound of the resulting diffusion operator, not
the maximum of scalar `D_k`.  Diffusive enthalpy transport must remain paired
with species flux; separate ion/electron heat flux and Dufour terms require an
explicit energy-budget decision.

## Cold-to-warm DT is the target path

Prioritize open data and published models, with reproducible derivations and
source-specific validity ranges. Access to proprietary EOS or transport tables
is not a prerequisite for this work; unresolved gaps in open coverage must stay
explicit rather than being filled with an unvalidated extrapolation.

An ICF fuel calculation begins near 19 K in molecular solid or liquid DT, not
in a fully ionized plasma.  It must cross phase change, molecular rotation and
vibration, dissociation, atomic-fluid behavior, ionization, electron degeneracy
and strong coupling before a hot-plasma model applies.  The EOS must account
for latent, dissociation and ionization energies in the same total-energy
variable used by hydrodynamics.  The transport model must consume the EOS's
species and charge-state populations so that neutral-neutral, neutral-ion,
ion-ion and electron collisions exchange dominance continuously.  Mean charge
alone is generally insufficient for collision physics when several charge or
neutral states coexist.

Kerley's [*Equations of State for Hydrogen and Deuterium*](https://digital.library.unt.edu/ark:/67531/metadc886665/)
is an open primary
description of a wide-range chemical EOS with molecular and atomic solids, a
molecular/atomic fluid, ionization and an insulator-metal transition.  It is a
model and validation source, not a freely licensed ready-to-vendor table.
Hu et al.'s [first-principles deuterium EOS](https://doi.org/10.1103/PhysRevB.84.224109),
[open manuscript](https://arxiv.org/abs/1110.0001), spans 1.35 eV to 5.5 keV and documents ICF-
significant differences from older SESAME tables; it does not cover the 19 K
initial state.  Caillabet et al. demonstrate an ab-initio multiphase DT EOS in
an implosion initialized at 19 K in [Phys. Rev. Lett. 107, 115004 (2011)](https://doi.org/10.1103/PhysRevLett.107.115004),
[open manuscript](https://arxiv.org/abs/1105.5495).  These sources define cross-checks, while
actual LEOS/SESAME/IONMIX redistribution and interpolation rights must be
resolved table by table.

No one source above closes the full path.  Richardson, Leachman and Lemmon's
[fundamental D2 fluid EOS](https://doi.org/10.1063/1.4864752) covers the melting
line through 600 K and pressures through 2000 MPa and distinguishes ortho/para
states, but it is neither a solid-DT nor a warm-plasma EOS.  The improved
[first-principles deuterium EOS](https://doi.org/10.1103/PhysRevB.104.144104)
extends roughly 800 K--256 MK and 0.001--1600 g/cm3 with quantum ions, leaving a
gap to cryogenic initialization; substituting isotope masses does not establish
cold-mixture accuracy.  Hu et al.'s [first-principles thermal conductivity](https://doi.org/10.1103/PhysRevE.89.043105)
covers dense warm deuterium from
about 5000 K and explicitly shows why Spitzer transport fails in cool dense ICF
shells, but supplies no 19 K conductivity.  The open [NBS cryogenic hydrogen
data report](https://www.osti.gov/servlets/purl/6205719) compiles measurements
below 30 K and estimates some D2/DT/T2 properties; estimated entries must remain
flagged and cannot silently fill the solid-to-warm gap.  Until compatible DT
phase, EOS and transport evidence overlaps these ranges, H5a remains a coverage
gap rather than an interpolation task.

Hot-limit Spitzer--Harm conduction cannot be extended to cold DT by a flux
limiter.  Lee and More's [wide-range electron transport model](https://doi.org/10.1063/1.864744)
joins solid, liquid and plasma relaxation models, and Desjarlais documents
[metal-insulator-transition improvements](https://doi.org/10.1002/1521-3986(200103)41:2/3%3C267::AID-CTPP267%3E3.0.CO;2-P),
including electron-neutral collisions and ionization changes.  They are useful
baselines, not DT-specific accuracy guarantees.  Warm-dense DT conductivity
needs comparison with Kubo--Greenwood/QMD results and experiment over the
actual density-temperature path.

There must be no arbitrary temperature switch and no manual zeroing of
transport that freezes the initial fuel until a conduction or radiation front
arrives.  Model blending, if unavoidable, is based on common state variables
and overlapping validity regions, preserves positive entropy production, and
is included in the uncertainty assessment.

## Other HED materials and mixtures

Roadmap H5b extends the material coverage work to C, CH and CD. These are
separate qualification targets; a successful DT model does not validate them.
For each target, first record the intended composition, material form, initial
phase and reference density/temperature, followed by the path through heating
and compression. The inventory must distinguish the following evidence:

| Quantity | Required qualification |
|---|---|
| EOS/recovery | References, phases, inversion, derivatives, equilibrium assumptions |
| Populations/electron density | EOS/inventory consistency and collision-model sufficiency |
| Transport | Phases/channels, state range, independent coefficients, uncertainty/gaps |
| Opacity when used | Composition/populations, groups, compatibility with material state |
| Mixture/contact model | Mixing/equilibrium rules, independent energy/transport checks |

Do not interpret CH/CD as a choice of gas-phase molecular species solely from
the label. A material table needs its actual composition and phase documented;
replacing H masses by D masses is an estimate unless independently qualified for
the property and regime concerned. Likewise, separate elemental EOS tables do
not by themselves specify a compound or a fuel–ablator mixture EOS. Declare
whether a calculation represents a homogeneous mixture or distinct materials
at an interface before selecting a transport closure.

This section establishes coverage requirements, not a claim that suitable C,
CH or CD datasets have been selected or bundled. Record source-specific evidence
and unresolved gaps here as H5b proceeds. Start each material's evolution tests
inside a validated range, then extend across transitions only when compatible
EOS, population and transport evidence supports them. The cold-to-warm DT
prohibitions on arbitrary switches and numerical freezing apply to these models
as well; physical suppression and numerical flux limiting remain distinct.

## Staged execution and gates

1. **H4a coefficient foundation, now.** Implement a standalone
   Stanton--Murillo evaluator for fully ionized `H+`, `D+`, and `T+` using the
   published masses, `Z = 1`, and Eq. 25 electron screening.  It returns binary interdiffusion and all
   dimensionless regime diagnostics required to judge the result; it does not
   enter the runtime flux.  Gate against archived outputs from the authors'
   evaluator, the published weak-coupling asymptote, exact interchange symmetry,
   dimensional analysis, and selected paper figures or molecular-dynamics
   points.  Test the isotope masses directly; do not test a post-hoc `H`
   rescaling.  Keep fully ionized, classical-ion, unmagnetized and equilibrium-
   temperature assumptions explicit, and report degeneracy even inside the
   current `theta >= 10` software domain.
2. **H5a cold-to-warm material contract.** Select an openly usable multiphase
   DT EOS or construct a documented table from open data and published models. Its query
   returns phase, molecular/atomic and charge-state populations (or a validated
   reduced population closure) and free electron density consistently with
   pressure, energy and derivatives. Pair
   it with a regime-dispatched conductivity/collision contract whose providers
   overlap; do not add a temperature switch or freeze state.  Gate table nodes,
   thermodynamic inversion and derivatives first, then a cold DT slab/contact
   heated by a prescribed boundary heat flux or incoming radiation.  Deposited
   energy immediately enters conserved material energy and a consistent
   phase/species state; temperature may remain on a physical latent-heat
   plateau but is never frozen numerically.  Measure the resulting heat front
   rather than prescribing its motion.  Total energy closes through
   phase/dissociation/ionization changes, fractions remain bounded, and
   refinement moves neither the measured transition location nor integrated
   energy outside tolerance.  Begin with a reference case wholly inside the
   validated cold-EOS range before extending it across evidence gaps.
3. **Thermodynamic-force foundation.** Implement Kagan--Tang binary driving
   forces as independently callable algebra: concentration, baro-, ion-thermo-,
   electron-thermo-, and electro-diffusion terms, without coupling them to the
   conserved flux.  Gate each force separately.  In particular, at zero
   concentration gradient require nonzero barodiffusion under a pressure
   gradient and nonzero thermodiffusion under ion/electron temperature
   gradients; require every force to vanish at uniform equilibrium.  Reproduce
   the published D--T and D--He3 sign/ratio cases, including the stated
   charge-to-mass cancellation limit.  This stage proves force algebra but makes
   no ambipolar or energy-coupling claim.
4. **N8a neutral data foundation.** Add a standalone, immutable pair-polynomial
   data type without changing the legacy `BinaryDiffusion`.  Ingest a small
   source-faithful `H2/HD/D2` fixture only after two-person transcription from
   the primary table.  Keep tritiated molecules out until their table and
   uncertainty are independently recovered.  Gate on symmetry, positivity,
   units, exact `1/p`, fit residuals at every source node, source-range behavior,
   and comparison with an independent published value not used by the fit.
   Delivered: `MARRERO_MASON_1972` carries Tables 12 and 13 of Marrero and
   Mason in SI after two independent transcriptions were diffed, with the
   H2-D2 fit gated against the paper's Table 20 nodes; `SONG_WANG_2016`
   carries the calculated isotopologue and helium pairs as fits made by
   `data/songwang_extract.jl` from the publisher's PDF and independently
   recovered by the Xpdf/layout audit in `data/songwang_verify.jl`, which
   checks all 78 mixture tables and 702 equimolar nodes; and
   `MUELLER_KLEMM_1970` carries the nine measured room-temperature values.
   `neutral_binary_diffusion` fits the polynomial type to any pair of them
   under a `source` preference list that keeps the three kinds distinct.
   The H2-D2 correlation, every calculated pair and every measurement
   agree within their stated uncertainties.  Tritiated pairs are included
   with their stated 2% error.
5. **Neutral flux integration.** Use the polynomial pairs in the existing
   mixture-averaged closure.  Gate with binary Loschmidt diffusion at uniform
   pressure and temperature, convergence, conservation, permutation of species
   order, and agreement between direct binary and mixture-averaged limits.
   Thermal diffusion is deliberately absent and must be stated in results.
6. **Plasma cross-model verification, offline.** Implement Paquette or an
   independently formulated effective-potential evaluator as a second
   coefficient model.  Compare its weak-coupling asymptote with
   Stanton--Murillo and selected points with an independent molecular-dynamics
   source.  Every result reports coupling, screening, degeneracy and ionization
   regime.  Disagreement is an uncertainty band, not a tuning target.
7. **State and flux infrastructure.** After roadmap H3 supplies `T_ele` and
   electron pressure, add charge-state metadata, electron momentum/current
   closure, gradients of ion/electron temperature and pressure, and the
   constrained multicomponent flux solve.  H1--H2 are required if the new
   diffusion or equilibration is stiff; H5 or an equivalent ionization/EOS
   source is required for partially ionized target states.
8. **Coupled HEDP transport.** Add strong-coupling model selection with explicit
   validity bounds, electron conduction consistently with H4, and neutral-ion
   transitions only after validated partial-ionization collision data exist.
   Gate the coefficient provider separately from the flux solve.  The provider
   gate covers published coefficients, symmetry, limits and model validity; the
   flux gate covers Onsager symmetry where applicable, mass and charge
   constraints, zero-current/ambipolar closure, equilibrium/no-force limits,
   nonzero baro/thermo separation at zero concentration gradient, and published
   baro/thermo/electrodiffusion tests.  The energy-coupling gate separately
   closes diffusive enthalpy, ion/electron heat and equilibration budgets before
   coupled implosions.  Cross-model/MD uncertainty bands accompany all three.

The first two stages are executable HEDP-first foundations, but coefficient and
material-contract tests alone are not plasma flux closure.  The neutral stages improve a
separate bounded capability.  Runtime plasma transport follows the state,
implicit-integration and EOS dependencies in roadmap H1--H5 rather than
bypassing them.
