# Boundary conditions.
#
# Grids are node-centered: closed dimensions include their endpoints
# (h = L/(N−1)), so wall states are enforced directly on the wall-plane nodes.
# `apply_bcs!` runs at the head of every Runge-Kutta stage and once more after
# the final update. Periodic dimensions omit the duplicate endpoint (h = L/N);
# a dimension carrying a coordinate fold is half-offset instead, so no node
# sits on the singular set and no state is enforced there.
#
# Spatial closure near boundaries is handled by the scheme's ClosureRows (see
# kernels.jl). The BC objects here declare periodicity to the decomposition
# through `isperiodic`, enforce state values on wall planes through `enforce!`,
# and correct the right-hand side through `correct_rhs!`. A new condition is a
# subtype of BoundaryCondition defining whichever of `enforce!(bc, Q, solver,
# dim, side)` and `correct_rhs!(bc, solver, Q, dQ, dim, side)` it needs, plus
# `validate_bc` when its derivation restricts the geometry or the EOS.

"""
    BoundaryCondition

Abstract supertype for physical-face conditions. A new condition implements
hard state enforcement with `enforce!`, an RHS characteristic correction with
`correct_rhs!`, or both. Periodic and coordinate-fold behavior must also be
declared during solver setup.
"""
abstract type BoundaryCondition end

"""Periodic continuation through the opposite face of one dimension."""
struct PeriodicBC <: BoundaryCondition end

"""
    SlipWallBC()

Impermeable inviscid wall: remove normal velocity while retaining tangential
velocity. The grid is node-centered and includes the wall point. On the wall
plane the normal momentum is set to zero and the total energy is reduced by the
normal kinetic energy it carried, so the internal energy is unchanged.
"""
struct SlipWallBC <: BoundaryCondition end

"""
    NoSlipWallBC(; Twall=NaN)

Impermeable viscous wall with zero velocity. All three momentum components on
the wall plane are set to zero and the total energy is reduced by the kinetic
energy they carried. The default non-finite `Twall` selects an adiabatic wall,
which stops there; a finite value selects an isothermal wall and overwrites the
total energy with the internal energy the EOS gives at `Twall`.
"""
struct NoSlipWallBC{T<:AbstractFloat} <: BoundaryCondition
    Twall::T   # NaN → adiabatic; finite → isothermal wall
end

NoSlipWallBC(; Twall::Real=NaN) = NoSlipWallBC(float(Twall))

"""
    ExtrapolationBC()

Copy the adjacent interior state onto the physical boundary plane. This simple
zero-normal-gradient approximation is not a characteristic non-reflecting
condition and should be justified for the intended flow.
"""
struct ExtrapolationBC <: BoundaryCondition end

"""
    AxisBC()

Regularized cylindrical axis at the low end of dimension 1: half-offset grid
(no node at r = 0) with parity mirror conditions. Requires
`CylindricalMetric`, an r origin at zero, and an unstretched r dimension. θ may
be either collapsed (axisymmetric, where each radial line continues into
itself) or resolved over 2π with an even point count, where it continues into
its antipodal partner. See folds.jl for the signs and the parallel-layout
restrictions.

There is no wall-plane state enforcement: the parity fill and the folded
implicit rows carry the whole treatment.
"""
struct AxisBC <: BoundaryCondition end

"""
    OriginBC()

Regularized spherical origin at the low end of r: half-offset grid plus the
antipodal fold (−r, θ, φ) ≡ (r, π−θ, φ+π). Requires `SphericalMetric`, an r
origin at zero, an unstretched r dimension, a θ range symmetric about π/2, and
φ collapsed or spanning 2π with an even point count. See folds.jl for signs and
parallel-layout restrictions.
"""
struct OriginBC <: BoundaryCondition end

"""
    PoleBC()

Regularized spherical polar axis: apply at BOTH ends of θ over (0, π) with a
half-offset θ grid; the fold pairs (−θ, φ) ≡ (θ, φ+π). Requires
`SphericalMetric`, an unstretched θ dimension, and φ collapsed or spanning 2π
with an even point count. Setup errors if it is applied at only one end of θ.
"""
struct PoleBC <: BoundaryCondition end

"""
    InterfaceBC(neighbor)

Marker condition on a patch face abutting another patch at the same level.
There is no state to enforce and no RHS correction: the coupling runs through
the interface ghost exchange, the interface closure rows, and the shared-plane
averaging (see patches.jl). `Solver` substitutes this onto interface faces
itself; it is not a user-supplied condition. `neighbor` is the abutting patch
id.
"""
struct InterfaceBC <: BoundaryCondition
    neighbor::Int
end

enforce!(::InterfaceBC, Q, solver, d, side) = nothing

"""
    CoarseFineBC()

Marker condition on a fine-patch face abutting the coarse level below it
(levels.jl). As with [`InterfaceBC`](@ref) there is nothing to enforce here:
the face's ghost layers and its boundary plane are overwritten from the
prolonged coarse state after every RK stage, and the line solves close with
the same extended-data rows a same-level interface uses. `Solver` substitutes
this onto refined-patch faces itself.
"""
struct CoarseFineBC <: BoundaryCondition end

enforce!(::CoarseFineBC, Q, solver, d, side) = nothing

isperiodic(::BoundaryCondition) = false
isperiodic(::PeriodicBC) = true

"""
    SwitchableBC(before, after)

One face that behaves as `before` until [`switch!`](@ref) is called on it, then
as `after`. This supports calculations that require one boundary condition
during an interaction and another to transmit the resulting outgoing waves.

The wrapper preserves the concrete type of `solver.bcs`; reassigning that field
to a different boundary-condition type would fail conversion.

Every rank must switch on the same step. `after` may perform collectives that
`before` does not; `NSCBCOutflowBC` is one example. Rank disagreement therefore
causes a collective-ordering deadlock. Use a [`Callback`](@ref), whose trigger
verdict is globally consistent, not a rank-local test.

Both conditions must agree on periodicity because `setup` uses that property to
construct the decomposition and line plans. Fold conditions (`AxisBC`,
`OriginBC`, and `PoleBC`) cannot be wrapped because `setup` identifies them by
the boundary-condition type itself.
"""
mutable struct SwitchableBC{B1<:BoundaryCondition,B2<:BoundaryCondition} <: BoundaryCondition
    before::B1
    after::B2
    switched::Bool
end

_is_fold_bc(bc) = bc isa AxisBC || bc isa OriginBC || bc isa PoleBC

function SwitchableBC(before::BoundaryCondition, after::BoundaryCondition)
    isperiodic(before) == isperiodic(after) ||
        throw(ArgumentError("SwitchableBC: both conditions must agree on periodicity"))
    (_is_fold_bc(before) || _is_fold_bc(after)) &&
        throw(ArgumentError("SwitchableBC cannot wrap a fold condition " *
                            "(AxisBC, OriginBC, PoleBC); setup detects those by type"))
    return SwitchableBC(before, after, false)
end

"""
    switch!(bc)

Select the `after` condition of a [`SwitchableBC`](@ref). Repeated calls have no
additional effect.
"""
switch!(bc::SwitchableBC) = (bc.switched = true; bc)

"Whether a [`SwitchableBC`](@ref) has switched yet."
switched(bc::SwitchableBC) = bc.switched

isperiodic(bc::SwitchableBC) = isperiodic(bc.before)

# Branch at the call site. Returning the active condition from a helper would
# make `active(bc)` return a small Union and widen both arms; here each arm is
# a concrete call.
enforce!(bc::SwitchableBC, Q, solver, d, side) =
    bc.switched ? enforce!(bc.after, Q, solver, d, side) :
                  enforce!(bc.before, Q, solver, d, side)

"Concrete plane type, so `for I in wallplane(...)` yields `CartesianIndex{3}`,
not `Any`. See the note on the constructor below."
const WallPlane = CartesianIndices{3,Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int}}}

"""
    wallplane(decomp, d, side) -> Union{Nothing,WallPlane}

Halo-offset `CartesianIndices` of the boundary plane on `side` of dimension `d`,
`side` being 1 for the low end and 2 for the high end, or `nothing` when this
rank does not own that global edge. The plane spans the full local extent of the
other two dimensions and is one point thick in `d`.

Ownership along `d` is the only test applied, so a periodic dimension yields a
plane as well; `apply_bcs!` and the `correct_rhs!` loop skip collapsed
dimensions before calling.
"""
function wallplane(decomp::Decomp, d::Int, side::Int)::Union{Nothing,WallPlane}
    if side == 1
        decomp.sub_rank[d] == 0 || return nothing
        i = 1
    else
        decomp.sub_rank[d] == decomp.sub_size[d] - 1 || return nothing
        i = decomp.n_local[d]
    end
    # Built out of three explicit locals, not ntuple(closure, 3):
    # inference widens the closure form to CartesianIndices{3,<:Tuple{
    # OrdinalRange,...}} even though every runtime value is a UnitRange, which
    # makes the loop variable in `for I in plane` infer as Any. Every array access
    # in the wall and NSCBC loops then goes through runtime dispatch, O(N^2)
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

"""
    correct_rhs!(bc, solver, Q, dQ, d, side)

RHS-level boundary hook, called once per face of every active dimension near the
end of `compute_rhs!`, after the metric sources and before the explicit sources.
An implementation adds its correction to `dQ` in place. The default is no
correction; the characteristic conditions in nscbc.jl override it.

Every rank calls this for every face, including ranks owning no part of the
plane. An implementation needing a distributed operator, `deriv_along!` among
them, must therefore issue it before testing `wallplane` for `nothing` and
returning; a collective below that return deadlocks as soon as the
boundary-normal dimension is decomposed. Both methods in nscbc.jl are written in
that order.
"""
correct_rhs!(bc::BoundaryCondition, solver, Q, dQ, d, side) = nothing

correct_rhs!(bc::SwitchableBC, solver, Q, dQ, d, side) =
    bc.switched ? correct_rhs!(bc.after, solver, Q, dQ, d, side) :
                  correct_rhs!(bc.before, solver, Q, dQ, d, side)

"""
    validate_bc(bc, metric, eos, d, side)

Setup-time hook, called by the [`Solver`](@ref) constructor once per face. The
default accepts anything. A condition whose derivation restricts the geometry, or
whose keywords must agree with the EOS, validates that here. The run then fails at
`setup`, avoiding a repeated check on every RHS call and preventing an uncovered
face from completing with a wrong answer.

Both arms of a [`SwitchableBC`](@ref) are validated, since the `after` condition
is reached without passing through setup again.
"""
validate_bc(::BoundaryCondition, metric, eos, d::Int, side::Int) = nothing

validate_bc(bc::SwitchableBC, metric, eos, d::Int, side::Int) =
    (validate_bc(bc.before, metric, eos, d, side);
     validate_bc(bc.after, metric, eos, d, side))

"ρe at a wall held at temperature Twall, one method per EOS."
wall_internal_energy(eos::IdealMixture, Q, I, n_species::Int, Twall) = begin
    ρe = zero(eltype(Q))
    @inbounds for k in 1:n_species
        ρe += Q[I, k] * eos.cvk[k]
    end
    ρe * Twall
end

# e = c_v T + p∞/ρ, so ρe picks up the cohesive term as a constant.
wall_internal_energy(eos::StiffenedGas, Q, I, ::Int, Twall) =
    @inbounds Q[I, 1] * eos.cv * Twall + eos.p_inf

# The device coefficient mirrors (physics.jl) reach here when the no-slip
# body runs as a kernel; same algebra as the host objects above.
wall_internal_energy(eos::IdealMixtureCoeffs, Q, I, n_species::Int, Twall) = begin
    ρe = zero(eltype(Q))
    @inbounds for k in 1:n_species
        ρe += Q[I, k] * eos.cvk[k]
    end
    ρe * Twall
end

wall_internal_energy(eos::StiffenedGasCoeffs, Q, I, ::Int, Twall) =
    @inbounds Q[I, 1] * eos.cv * Twall + eos.p_inf

wall_internal_energy(eos::Nasa9Mixture, Q, I, n_species::Int, Twall) = begin
    ρe = zero(eltype(Q))
    @inbounds for k in 1:n_species
        ρe += Q[I, k] * species_energy(eos, k, Twall)
    end
    ρe
end

function _wall_density(Q, I, n_species)
    ρ = zero(eltype(Q))
    @inbounds for k in 1:n_species
        ρ += Q[I, k]
    end
    return ρ
end

# Wall-plane enforcement runs as pointwise bodies over the plane's index
# box: one shared body per condition, launched by `plane_pointwise!`, so a
# device-resident patch enforces its walls without a host round trip. The
# per-point writes are independent, so the launch reproduces the former serial
# plane loop bitwise on the host path too.

"""
    plane_pointwise!(body!, route, plane, args...)

Launch `body!` through [`pointwise!`](@ref) over the padded index box of a
[`wallplane`](@ref) result, appending the plane's index offsets in the
`(o1, o2, o3)` slots the pointwise bodies use. The caller has tested
`plane` for `nothing`.
"""
@inline function plane_pointwise!(body!::F, route, plane::WallPlane,
                                  args...) where {F}
    r1, r2, r3 = plane.indices
    return pointwise!(body!, route, length(r1), length(r2), length(r3),
                      args..., first(r1) - 1, first(r2) - 1, first(r3) - 1)
end

@inline function _slip_wall_point!(Q, mc, i_energy, n_species, o1, o2, o3,
                                   i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = _wall_density(Q, I, n_species)
        mn = Q[I, mc]
        Q[I, i_energy] -= mn * mn / (oftype(ρ, 2) * ρ)
        Q[I, mc] = 0
    end
    return nothing
end

function enforce!(::SlipWallBC, Q, solver, d, side)
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    plane_pointwise!(_slip_wall_point!, Q, plane, Q, solver.equations.i_mom[d],
                     solver.equations.i_energy, solver.equations.n_species)
    nothing
end

@inline function _no_slip_wall_point!(Q, eos, Twall, iso, m1, m2, m3, i_energy,
                                      n_species, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = _wall_density(Q, I, n_species)
        ke = (Q[I,m1]^2 + Q[I,m2]^2 + Q[I,m3]^2) /
             (oftype(ρ, 2) * ρ)
        Q[I, m1] = 0
        Q[I, m2] = 0
        Q[I, m3] = 0
        # Adiabatic: the kinetic energy leaves with the momentum. Isothermal:
        # the wall temperature sets the internal energy outright.
        Q[I, i_energy] = iso ?
            wall_internal_energy(eos, Q, I, n_species, Twall) :
            Q[I, i_energy] - ke
    end
    return nothing
end

function enforce!(bc::NoSlipWallBC, Q, solver, d, side)
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    m1, m2, m3 = solver.equations.i_mom
    plane_pointwise!(_no_slip_wall_point!, Q, plane, Q, solver.eos, bc.Twall,
                     !isnan(bc.Twall), m1, m2, m3, solver.equations.i_energy,
                     solver.equations.n_species)
    nothing
end

@inline function _extrapolation_point!(Q, s1, s2, s3, n_cons, o1, o2, o3,
                                       i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        J = CartesianIndex(i + o1 + s1, j + o2 + s2, k + o3 + s3)
        for c in 1:n_cons
            Q[I, c] = Q[J, c]
        end
    end
    return nothing
end

function enforce!(::ExtrapolationBC, Q, solver, d, side)
    # Zeroth-order extrapolation: copy the adjacent interior plane onto the
    # edge plane. Crude outflow; NSCBCOutflowBC is the characteristic
    # alternative when reflections affect the solution.
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    sgn = side == 1 ? 1 : -1
    s = ntuple(k -> k == d ? sgn : 0, 3)
    plane_pointwise!(_extrapolation_point!, Q, plane, Q, s[1], s[2], s[3],
                     solver.equations.n_cons)
    nothing
end

"Enforce every boundary condition on the conserved state `Q` in place, over the
active dimensions only, and return `Q`. No condition CompactLES provides
communicates here; each acts through `wallplane` and so does nothing on a rank
owning no part of the face."
function apply_bcs!(solver, Q)
    for d in 1:3, side in 1:2
        solver.decomp.active[d] || continue
        enforce!(solver.bcs[d][side], Q, solver, d, side)
    end
    return Q
end
