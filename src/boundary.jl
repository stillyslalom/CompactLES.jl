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
# `enforce!(bc, Q, s, dim, side)`.

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

"CartesianIndices of the wall plane for (dim, side), or nothing if this rank
does not own that global edge."
function wallplane(dec::Decomp, d::Int, side::Int)
    if side == 1
        dec.subrank[d] == 0 || return nothing
        i = 1
    else
        dec.subrank[d] == dec.subsize[d] - 1 || return nothing
        i = dec.nloc[d]
    end
    Hd = dec.Hd
    CartesianIndices(ntuple(k -> k == d ? (Hd[k]+i:Hd[k]+i) :
                                          (Hd[k]+1:Hd[k]+dec.nloc[k]), 3))
end

enforce!(::PeriodicBC, Q, s, d, side) = nothing
enforce!(::AxisBC, Q, s, d, side) = nothing
enforce!(::OriginBC, Q, s, d, side) = nothing
enforce!(::PoleBC, Q, s, d, side) = nothing

"RHS-level boundary hook, called at the end of compute_rhs! for every face.
Default: no correction. Characteristic conditions (nscbc.jl) override this."
correct_rhs!(bc::BoundaryCondition, s, Q, dQ, d, side) = nothing

"ρe at a wall held at temperature Twall (EOS barrier; ideal mixtures)."
wall_internal_energy(eos::IdealMixture, Q, I, ns::Int, Twall) = begin
    ρe = 0.0
    @inbounds for k in 1:ns
        ρe += Q[I, k] * eos.cvk[k]
    end
    ρe * Twall
end

function _wall_density(Q, I, ns)
    ρ = 0.0
    @inbounds for k in 1:ns
        ρ += Q[I, k]
    end
    return ρ
end

function enforce!(::SlipWallBC, Q, s, d, side)
    pl = wallplane(s.dec, d, side)
    pl === nothing && return nothing
    mc = s.mom[d]
    @inbounds for I in pl
        ρ = _wall_density(Q, I, s.ns)
        mn = Q[I, mc]
        Q[I, s.ie] -= 0.5 * mn * mn / ρ   # remove normal kinetic energy
        Q[I, mc] = 0
    end
    nothing
end

function enforce!(bc::NoSlipWallBC, Q, s, d, side)
    pl = wallplane(s.dec, d, side)
    pl === nothing && return nothing
    m1, m2, m3 = s.mom
    iso = !isnan(bc.Twall)
    @inbounds for I in pl
        ρ = _wall_density(Q, I, s.ns)
        ke = 0.5 * (Q[I,m1]^2 + Q[I,m2]^2 + Q[I,m3]^2) / ρ
        Q[I, s.ie] -= ke
        Q[I, m1] = 0
        Q[I, m2] = 0
        Q[I, m3] = 0
        if iso
            Q[I, s.ie] = wall_internal_energy(s.eos, Q, I, s.ns, bc.Twall)
        end
    end
    nothing
end

function enforce!(::ExtrapolationBC, Q, s, d, side)
    # Zeroth-order extrapolation: copy the adjacent interior plane onto the
    # edge plane. Crude outflow; characteristic (NSCBC) treatment is a TODO.
    pl = wallplane(s.dec, d, side)
    pl === nothing && return nothing
    e = CartesianIndex(ntuple(k -> k == d ? 1 : 0, 3))
    shift = side == 1 ? e : -e
    @inbounds for I in pl, c in 1:s.ncons
        Q[I, c] = Q[I + shift, c]
    end
    nothing
end

"Enforce all boundary conditions on the conserved state (active dims only)."
function apply_bcs!(s, Q)
    for d in 1:3, side in 1:2
        s.dec.active[d] || continue
        enforce!(s.bcs[d][side], Q, s, d, side)
    end
    return Q
end
