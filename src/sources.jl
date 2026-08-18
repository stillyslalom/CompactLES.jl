# Explicit source-term interface. Source collections are tuples so every
# source type remains visible to inference and the recursion compiles away.

"""
    ConstantBodyForce(acceleration)
    ConstantBodyForce(; acceleration=(0.0, -9.81, 0.0))

Uniform body acceleration with three physical components in the
coordinate-aligned orthonormal basis. At each interior point it adds
`rho * acceleration` to momentum and `rho * dot(u, acceleration)` to total
energy, with `rho` the mixture density summed over the species entries of `Q`.

The positional form accepts a three-tuple of real values and promotes them to
a common numeric type. The keyword form defaults to a gravity-like acceleration
in the negative second direction. Values must use the length/time-squared unit
consistent with the rest of the problem.
"""
struct ConstantBodyForce{T}
    acceleration::NTuple{3,T}
end

function ConstantBodyForce(acceleration::NTuple{3,<:Real})
    values = promote(acceleration...)
    return ConstantBodyForce{typeof(values[1])}(values)
end

ConstantBodyForce(; acceleration=(0.0, -9.81, 0.0)) =
    ConstantBodyForce(acceleration)

@inline function _body_force_point!(dQ, Q, g1, g2, g3, n_species,
                                    m1, m2, m3, i_energy, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        rho = zero(eltype(Q))
        for sp in 1:n_species
            rho += Q[I, sp]
        end
        dQ[I, m1] += rho * g1
        dQ[I, m2] += rho * g2
        dQ[I, m3] += rho * g3
        dQ[I, i_energy] += Q[I, m1] * g1 + Q[I, m2] * g2 + Q[I, m3] * g3
    end
    return nothing
end

"""
    add_source!(source, solver, dQ, Q, t)

Add one explicit source to the already assembled conserved RHS at Runge-Kutta
stage time `t`. `Q` and `dQ` are halo-padded conserved arrays indexed
`[I, component]`, with the component layout defined by `solver.equations`.
Extend this function for a custom concrete source type and place instances in
`Problem.sources`. The method runs on every rank and must preserve that
component layout.

A custom method mutates only the rank-local interior of `dQ`, adds rather than
overwrites contributions already present, and returns `dQ`. It may inspect `Q`
and `solver`; any collective operation must be entered by every rank.
"""
function add_source!(source::ConstantBodyForce, solver, dQ, Q, t)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    g1, g2, g3 = source.acceleration
    pointwise!(_body_force_point!, dQ, nx, ny, nz,
               dQ, Q, g1, g2, g3, n_species, m1, m2, m3, i_energy, o1, o2, o3)
    return dQ
end

@inline _add_sources!(::Tuple{}, solver, dQ, Q, t) = dQ

@inline function _add_sources!(sources::Tuple, solver, dQ, Q, t)
    add_source!(first(sources), solver, dQ, Q, t)
    return _add_sources!(Base.tail(sources), solver, dQ, Q, t)
end

"""
    add_sources!(solver, dQ, Q, t)

Apply every source in `solver.sources`, in tuple order, to the already assembled
RHS `dQ` at stage time `t`. An empty tuple is a no-op. The function mutates and
returns `dQ`; users normally call it indirectly through [`compute_rhs!`](@ref).
"""
add_sources!(solver, dQ, Q, t) =
    _add_sources!(solver.sources, solver, dQ, Q, t)
