# Bench-only detector splits for a run built under `detector = :d8`, included
# by bench/falseactivation.jl and bench/he_co2_shock_tube.jl. The solver has
# one detector for every sensed field; these redefine its dispatch so that
# some fields take `:d8` and the rest δ⁴. Nothing in src offers either split.
#
#   SPLIT[]         `:d8` on the fields of the κ* and D* channels (the internal
#                   energy, the mass and the mole fractions), δ⁴ on |S| and
#                   ∇·u, the fields of μ* and β*: a weight power of 2, which
#                   only those two sensors pass, goes to `delta4_sum!`.
#   SPECIES_ONLY[]  `:d8` on the mass and mole fractions only, δ⁴ on every
#                   other field. The mass and mole fractions are sensed only
#                   inside `bulk_diffusivity!` under the default species
#                   channel, which the redefined `_bulk_diffusivity!` below
#                   brackets with `IN_SPECIES[]`; a run without species never
#                   reaches it and is the δ⁴ run bit for bit. The `:fickian`
#                   channel senses its mass fractions elsewhere and is not
#                   split.
#
# Include after `using CompactLES`; neither flag calls MPI.

using CompactLES
const SPLIT = Ref(false)
const SPECIES_ONLY = Ref(false)
const IN_SPECIES = Ref(false)

function CompactLES._detect_sum!(out, f, solver, wpow::Int, acc::Bool, par, wpar,
                                 gh::Bool, ::Tuple)
    if (SPLIT[] && wpow == 2) || (SPECIES_ONLY[] && !IN_SPECIES[])
        return CompactLES.delta4_sum!(out, f, solver, wpow; accumulate=acc, parity=par,
                                      wall_parity=wpar, ghosts=gh)
    end
    return CompactLES.ring_sum!(out, f, solver, wpow; accumulate=acc, parity=par,
                                wall_parity=wpar, ghosts=gh)
end

# The src method with the species bracket added.
function CompactLES._bulk_diffusivity!(solver)
    art = solver.art
    h_bound, inv_n = CompactLES._species_bound_length(solver.decomp, solver.h, art.C_D)
    a1, a2, a3 = solver.decomp.active
    ih1, ih2, ih3 = solver.inv_h
    IN_SPECIES[] = true
    try
        return CompactLES.bulk_diffusivity!(solver, art.C_D, art.C_Y, h_bound, inv_n,
                                            ih1, ih2, ih3, a1, a2, a3, art.Y_tolerance)
    finally
        IN_SPECIES[] = false
    end
end
