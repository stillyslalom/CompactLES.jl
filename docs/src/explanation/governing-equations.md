# Governing equations

CompactLES solves the compressible, multicomponent Navier--Stokes equations in
conservative form. This page fixes the notation, assumptions, and relation
between the mathematical state and the Julia objects used elsewhere in the
manual.

## Conserved state

For `Ns` species, the state stored at each grid point is

```math
Q = (\rho Y_1,\ldots,\rho Y_{N_s},
     \rho u_1,\rho u_2,\rho u_3,E)^T.
```

Here ``rho`` is mixture density, ``Y_k`` is a species mass fraction, ``u`` is
the physical velocity in the local coordinate basis, and ``E`` is total energy
per unit volume. The constraints are

```math
\rho = \sum_k \rho Y_k, \qquad \sum_k Y_k = 1,
```

and

```math
E = \rho e + \tfrac12 \rho |\boldsymbol{u}|^2,
```

where ``e`` is mixture specific internal energy. `Prim` presents the same state
in primitive variables; the selected EOS maps between the two representations.

## Conservation laws

Without chemical reactions, each species satisfies

```math
\frac{\partial \rho Y_k}{\partial t}
+ \nabla\!\cdot(\rho Y_k\boldsymbol{u} + \boldsymbol{J}_k) = 0.
```

Momentum satisfies

```math
\frac{\partial \rho\boldsymbol{u}}{\partial t}
+ \nabla\!\cdot
  (\rho\boldsymbol{u}\otimes\boldsymbol{u} + p\boldsymbol{I}
   - \boldsymbol{\tau})
= \rho\boldsymbol{f},
```

and total energy satisfies

```math
\frac{\partial E}{\partial t}
+ \nabla\!\cdot\left[
  (E+p)\boldsymbol{u}
  - \boldsymbol{\tau}\!\cdot\boldsymbol{u}
  + \boldsymbol{q}
  + \sum_k h_k\boldsymbol{J}_k\right]
= \rho\boldsymbol{f}\!\cdot\boldsymbol{u}.
```

`ConstantBodyForce` supplies ``f``. Additional sources enter the same
conservative right-hand side through `Problem.sources`.

The viscous stress convention is

```math
\boldsymbol{\tau}
= \mu\left[\nabla\boldsymbol{u}+(\nabla\boldsymbol{u})^T
-\frac{2}{3}(\nabla\!\cdot\boldsymbol{u})\boldsymbol{I}\right]
+\beta(\nabla\!\cdot\boldsymbol{u})\boldsymbol{I}.
```

Here ``\mu`` contains molecular and artificial shear viscosity. The present
model has no molecular bulk viscosity; ``\beta`` is the artificial bulk
viscosity used to spread shocks.

## Molecular and artificial transport

Molecular viscosity ``\mu_0`` is presently constant. The molecular heat
conductivity and species diffusivity are

```math
\kappa_0=\frac{\mu_0c_p}{\mathrm{Pr}},\qquad
D_0=\frac{\mu_0}{\rho\,\mathrm{Sc}}.
```

The conductive heat flux is

```math
\boldsymbol{q} = -\kappa\nabla T,
```

and the species flux uses a correction velocity:

```math
\boldsymbol{J}_k = -\rho D_k\nabla Y_k
+ \rho Y_k\sum_j D_j\nabla Y_j.
```

The second term makes ``sum(J_k)=0`` exactly, so species diffusion cannot create
or destroy mixture mass. Species enthalpy transport `sum(h_k J_k)` is included
in the energy equation.

Artificial shear viscosity, bulk viscosity, conductivity, and species
diffusivity add to their molecular counterparts near under-resolved gradients.
They are numerical regularization, not constitutive properties of the fluid;
see [Filtering and artificial properties](@ref).

## Thermodynamic closure

The conservation laws do not determine pressure and temperature without an
equation of state. CompactLES currently supplies:

- a calorically perfect ideal mixture;
- an ideal mixture with NASA-9 temperature-dependent heat capacities; and
- a single-component stiffened gas.

The EOS provides pressure, temperature, sound speed, mixture heat capacity, and
species enthalpies from the conserved state. See
[Thermodynamics and species transport](@ref) for their assumptions.

## Geometry

The same conservation laws are evaluated in Cartesian, cylindrical, or
spherical coordinates. Stored velocity components are physical components in
the local orthonormal basis, not derivatives of the coordinate values. Metric
area factors enter conservative flux divergence, and rotating-basis terms enter
velocity gradients and momentum sources. See
[Curvilinear coordinates](@ref).

## From the equations to a numerical right-hand side

After the grid and spatial operators are fixed, the nodal conserved state
``Q_h`` satisfies a finite system of ordinary differential equations,

```math
\frac{dQ_h}{dt}=R_h(Q_h,t).
```

For every evaluation of ``R_h``, CompactLES recovers primitive and
thermodynamic variables from ``Q_h``, differentiates velocity, temperature,
and composition, constructs the complete physical and artificial fluxes,
takes their metric-weighted divergence, and then adds geometric terms,
boundary corrections, and explicit sources. The
[Spatial and temporal discretization](@ref) page explains both that spatial
operator and how repeated evaluations of ``R_h`` advance one timestep.

## Units and nondimensionalization

CompactLES attaches no unit type to a number. Inputs may be SI or consistently
nondimensional. Every quantity in one problem must use the same system,
including EOS gas constants, pressure, density, temperature, viscosity, source
terms, domain lengths, and output times.

Dimensionless examples commonly select reference values such that background
density, pressure, and length are one. Dimensional thermochemistry from the
bundled NASA database uses SI-specific gas constants, so pressure in pascals,
density in kilograms per cubic metre, and temperature in kelvin form the
natural accompanying convention.

## Model scope

The current equations omit reactions, radiation, ionization, external magnetic
fields, and multiphase interface physics. There is one temperature, named
`T_ion` to reserve `T_ele` and `T_rad` for possible future models. A numerical
solution is evidence about this closed model, not automatically about a physical
experiment; the configuration must be validated for its intended regime.
