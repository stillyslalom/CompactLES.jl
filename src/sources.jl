# Explicit source-term interface. Source collections are tuples so every
# source type remains visible to inference and the recursion compiles away.

"""
    ConstantBodyForce(acceleration)

Uniform acceleration. Adds rho*g to momentum and (rho*u) dot g to total
energy at every interior point.
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

function add_source!(source::ConstantBodyForce, solver, dQ, Q, t)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    g1, g2, g3 = source.acceleration
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
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
    end
    return dQ
end

@inline _add_sources!(::Tuple{}, solver, dQ, Q, t) = dQ

@inline function _add_sources!(sources::Tuple, solver, dQ, Q, t)
    add_source!(first(sources), solver, dQ, Q, t)
    return _add_sources!(Base.tail(sources), solver, dQ, Q, t)
end

"Apply every source in `solver.sources` to the already assembled RHS."
add_sources!(solver, dQ, Q, t) =
    _add_sources!(solver.sources, solver, dQ, Q, t)

