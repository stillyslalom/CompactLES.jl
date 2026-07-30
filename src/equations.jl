# Conserved-variable layouts. Equation sets own component indices, names, and
# the parity rules needed at coordinate-singularity folds. The spatial
# discretization consumes this interface without reconstructing a layout from
# the EOS species count.

"""
    EquationSet

Abstract interface for an equation set that owns the conserved-variable layout
and its parity across coordinate folds. [`NavierStokes1T`](@ref) is the current
implementation and is selected automatically by [`setup`](@ref).

A concrete equation set provides `n_species`, `n_cons`, `i_mom`, `i_energy`,
and `component_names`, together with methods for primitive-to-conserved
conversion and conserved-variable parity. Its species count must agree with
[`nspecies`](@ref) for the selected EOS.
"""
abstract type EquationSet end

"""
    NavierStokes1T(eos)

Construct the single-temperature, multicomponent Navier--Stokes equation set
for `eos`. Its conserved layout is
`(rho*Y[1], ..., rho*Y[Ns], rho*u1, rho*u2, rho*u3, rho*E)`: one partial
density per EOS species, three momentum components in the coordinate-aligned
orthonormal basis, and total energy per volume.

The object records the component indices and names used by I/O, source terms,
boundary conditions, and coordinate folds. Users normally select it implicitly
through [`setup`](@ref); construct it directly when inspecting a layout or
using the low-level [`Solver`](@ref) constructor.
"""
struct NavierStokes1T <: EquationSet
    n_species::Int
    n_cons::Int
    i_mom::NTuple{3,Int}
    i_energy::Int
    component_names::Vector{String}
end

species_names(eos::IdealMixture) = [sp.name for sp in eos.sp]
species_names(eos::EOS) = ["species_$k" for k in 1:nspecies(eos)]

function NavierStokes1T(eos::EOS)
    n_species = nspecies(eos)
    n_cons = n_species + 4          # partial densities, three momenta, energy
    i_mom = (n_species + 1, n_species + 2, n_species + 3)
    i_energy = n_species + 4        # equals n_cons only because energy is last
    names = ["rho_" * name for name in species_names(eos)]
    append!(names, ("rho_u1", "rho_u2", "rho_u3", "rho_E"))
    return NavierStokes1T(n_species, n_cons, i_mom, i_energy, names)
end

"Parity of conserved component `c` under the basis change `sigvel`."
@inline function conserved_parity(equations::NavierStokes1T, sigvel, c::Int)
    c <= equations.n_species && return 1
    c == equations.i_energy && return 1
    return sigvel[c - equations.n_species]
end

"Flux parity table for a fold along `dim`, including area-factor parity."
function flux_parities(equations::EquationSet, sigvel, dim::Int, area_parity::Int)
    normal_parity = sigvel[dim]
    return Int[conserved_parity(equations, sigvel, c) *
               normal_parity * area_parity for c in 1:equations.n_cons]
end
