# Boundary conditions.
#
# Grids are node-centered: closed dimensions include their endpoints
# (h = L/(N−1)), so wall states are enforced directly on the wall-plane nodes
# after each stage. Periodic dimensions omit the duplicate endpoint (h = L/N).
#
# Spatial closure near boundaries is handled by the scheme's ClosureRows (see
# kernels.jl); the BC objects here (a) declare periodicity to the decomposition
# and (b) enforce state values on wall planes via `enforce!`. New conditions
# are added by subtyping BoundaryCondition and defining
# `enforce!(bc, Q, solver, dim, side)`.

abstract type BoundaryCondition end

struct PeriodicBC <: BoundaryCondition end
struct SlipWallBC <: BoundaryCondition end
Base.@kwdef struct NoSlipWallBC <: BoundaryCondition
    Twall::Float64 = NaN   # NaN → adiabatic; finite → isothermal wall
end
struct ExtrapolationBC <: BoundaryCondition end

"""
    AxisBC()

Regularized cylindrical axis at the low end of dimension 1: half-offset grid
(no node at r = 0) with parity mirror conditions. Requires
`CylindricalMetric`, collapsed θ (axisymmetric), and an unstretched r
dimension. No wall-plane state enforcement is needed — the parity fill and
folded implicit rows carry the whole treatment.
"""
struct AxisBC <: BoundaryCondition end

"""
    OriginBC()

Regularized spherical origin at the low end of r: half-offset grid plus the
antipodal fold (−r, θ, φ) ≡ (r, π−θ, φ+π). Requires `SphericalMetric`, a θ
range symmetric about π/2, and φ collapsed or spanning 2π with an even point
count. See folds.jl for signs and parallel-layout restrictions.
"""
struct OriginBC <: BoundaryCondition end

"""
    PoleBC()

Regularized spherical polar axis: apply at BOTH ends of θ over (0, π) with a
half-offset θ grid; the fold pairs (−θ, φ) ≡ (θ, φ+π). Requires
`SphericalMetric` and φ collapsed or spanning 2π with an even point count.
"""
struct PoleBC <: BoundaryCondition end

isperiodic(::BoundaryCondition) = false
isperiodic(::PeriodicBC) = true

"Concrete plane type, so `for I in wallplane(...)` yields `CartesianIndex{3}`
rather than `Any`. See the note on the constructor below."
const WallPlane = CartesianIndices{3,Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int}}}

"CartesianIndices of the wall plane for (dim, side), or nothing if this rank
does not own that global edge."
function wallplane(decomp::Decomp, d::Int, side::Int)::Union{Nothing,WallPlane}
    if side == 1
        decomp.sub_rank[d] == 0 || return nothing
        i = 1
    else
        decomp.sub_rank[d] == decomp.sub_size[d] - 1 || return nothing
        i = decomp.n_local[d]
    end
    # Built out of three explicit locals rather than ntuple(closure, 3):
    # inference widens the closure form to CartesianIndices{3,<:Tuple{
    # OrdinalRange,...}} even though every runtime value is a UnitRange, which
    # makes the loop variable in `for I in plane` infer as Any. Every array access
    # in the wall and NSCBC loops then goes through runtime dispatch — O(N^2)
    # of them per face per RK stage.
    n_halo_d = decomp.n_halo_d
    n = decomp.n_local
    r1 = d == 1 ? (n_halo_d[1]+i:n_halo_d[1]+i) : (n_halo_d[1]+1:n_halo_d[1]+n[1])
    r2 = d == 2 ? (n_halo_d[2]+i:n_halo_d[2]+i) : (n_halo_d[2]+1:n_halo_d[2]+n[2])
    r3 = d == 3 ? (n_halo_d[3]+i:n_halo_d[3]+i) : (n_halo_d[3]+1:n_halo_d[3]+n[3])
    return CartesianIndices((r1, r2, r3))
end

enforce!(::PeriodicBC, Q, solver, d, side) = nothing
enforce!(::AxisBC, Q, solver, d, side) = nothing
enforce!(::OriginBC, Q, solver, d, side) = nothing
enforce!(::PoleBC, Q, solver, d, side) = nothing

"RHS-level boundary hook, called at the end of compute_rhs! for every face.
Default: no correction. Characteristic conditions (nscbc.jl) override this."
correct_rhs!(bc::BoundaryCondition, solver, Q, dQ, d, side) = nothing

"ρe at a wall held at temperature Twall (EOS barrier; ideal mixtures)."
wall_internal_energy(eos::IdealMixture, Q, I, n_species::Int, Twall) = begin
    ρe = 0.0
    @inbounds for k in 1:n_species
        ρe += Q[I, k] * eos.cvk[k]
    end
    ρe * Twall
end

function _wall_density(Q, I, n_species)
    ρ = 0.0
    @inbounds for k in 1:n_species
        ρ += Q[I, k]
    end
    return ρ
end

function enforce!(::SlipWallBC, Q, solver, d, side)
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    mc = solver.equations.i_mom[d]
    @inbounds for I in plane
        ρ = _wall_density(Q, I, solver.equations.n_species)
        mn = Q[I, mc]
        Q[I, solver.equations.i_energy] -= 0.5 * mn * mn / ρ   # remove normal kinetic energy
        Q[I, mc] = 0
    end
    nothing
end

function enforce!(bc::NoSlipWallBC, Q, solver, d, side)
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    m1, m2, m3 = solver.equations.i_mom
    iso = !isnan(bc.Twall)
    @inbounds for I in plane
        ρ = _wall_density(Q, I, solver.equations.n_species)
        ke = 0.5 * (Q[I,m1]^2 + Q[I,m2]^2 + Q[I,m3]^2) / ρ
        Q[I, solver.equations.i_energy] -= ke
        Q[I, m1] = 0
        Q[I, m2] = 0
        Q[I, m3] = 0
        if iso
            Q[I, solver.equations.i_energy] =
                wall_internal_energy(solver.eos, Q, I, solver.equations.n_species, bc.Twall)
        end
    end
    nothing
end

function enforce!(::ExtrapolationBC, Q, solver, d, side)
    # Zeroth-order extrapolation: copy the adjacent interior plane onto the
    # edge plane. Crude outflow; characteristic (NSCBC) treatment is a TODO.
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    e = CartesianIndex(ntuple(k -> k == d ? 1 : 0, 3))
    shift = side == 1 ? e : -e
    @inbounds for I in plane, c in 1:solver.equations.n_cons
        Q[I, c] = Q[I + shift, c]
    end
    nothing
end

"Enforce all boundary conditions on the conserved state (active dims only)."
function apply_bcs!(solver, Q)
    for d in 1:3, side in 1:2
        solver.decomp.active[d] || continue
        enforce!(solver.bcs[d][side], Q, solver, d, side)
    end
    return Q
end
