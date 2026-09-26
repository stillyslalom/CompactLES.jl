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
# impose assembled fluxes through `correct_flux!`, and correct the right-hand
# side through `correct_rhs!`. A new condition is a subtype of BoundaryCondition
# defining whichever of the state, flux, and RHS hooks it needs, plus
# `validate_bc` when its derivation restricts the geometry or the EOS.

"""
    BoundaryCondition

Abstract supertype for physical-face conditions. A new condition implements
hard state enforcement with `enforce!`, a pre-divergence flux condition with
`correct_flux!`, and/or an RHS characteristic correction with `correct_rhs!`.
Periodic and coordinate-fold behavior must also be declared during solver setup.
"""
abstract type BoundaryCondition end

"""
Storage type of a patch's six face conditions, `bcs[dim][side]`. The element
type is abstract on purpose: see the note on [`Patch`](@ref) for why the
conditions are kept out of that type's parameters, and what the resulting
dispatch in the boundary hooks costs.
"""
const FaceConditions = NTuple{3,Tuple{BoundaryCondition,BoundaryCondition}}

# Normalize deck shorthand once, before geometry checks or patch construction.
# A bare condition shares the supplied object across both faces; in particular,
# switching a shared SwitchableBC intentionally switches both faces together.
function _face_conditions(bcs)
    bcs isa Tuple && length(bcs) == 3 ||
        throw(ArgumentError("bcs must be a three-tuple of conditions or face pairs"))
    return ntuple(3) do d
        pair = bcs[d]
        pair isa BoundaryCondition && return (pair, pair)
        pair isa Tuple && length(pair) in (1, 2) ||
            throw(ArgumentError("bcs[$d] must be a boundary condition or a face pair"))
        all(bc -> bc isa BoundaryCondition, pair) ||
            throw(ArgumentError("bcs[$d] entries must be BoundaryCondition objects"))
        length(pair) == 1 ? (pair[1], pair[1]) : pair
    end
end

"""
    enforce!(bc, Q, solver, dim, side)

Hard-state boundary hook. It is called once for each face of every active
dimension at the beginning of a Runge--Kutta stage and may modify the boundary
plane of the conserved state `Q`. The default is a no-op, which is appropriate
for conditions implemented entirely through [`correct_rhs!`](@ref).

Extend this method for a custom [`BoundaryCondition`](@ref). `dim` is 1, 2, or
3 and `side` is 1 (low) or 2 (high); only the rank owning the physical face
receives a nonempty plane. Return `nothing` or `Q` consistently from custom
methods; the solver uses the mutation, not the return value.
"""
enforce!(::BoundaryCondition, Q, solver, dim, side) = nothing

"""Periodic continuation through the opposite face of one dimension."""
struct PeriodicBC <: BoundaryCondition end

"""
    SlipWallBC()

Impermeable adiabatic symmetry plane: remove normal velocity while retaining
tangential velocity. The grid is node-centered and includes the wall point. On
the wall plane the normal momentum is set to zero and the total energy is
reduced by the normal kinetic energy it carried, so the internal energy is
unchanged.

Every species has zero total normal flux, including molecular and artificial
diffusion, and so does the total energy: the wall is adiabatic, the vanishing
normal velocity does no work, and no enthalpy or artificial `:bulk` flux
crosses the plane. The tangential momentum fluxes are zero as well, since the
tangential velocity's normal derivative vanishes at a symmetry plane and there
is no shear traction. The normal momentum flux remains, carrying the pressure,
the normal viscous stress and the dilatational term. The condition is applied
to the fully assembled flux, before the exchange and the compact divergence,
and at corners as well as faces. Filtering and the hard state enforcement
above are separate operations; neither establishes a whole-domain discrete
conservation identity.
"""
struct SlipWallBC <: BoundaryCondition end

"""
    NoSlipWallBC(; Twall=NaN)

Stationary, impermeable, noncatalytic viscous wall. All three momentum components
on the wall plane are set to zero. With the default `Twall=NaN`, the kinetic
energy they carried is removed and the total normal energy flux is set to zero
before divergence (adiabatic). A finite `Twall` instead sets the internal energy
from the EOS at that temperature and permits the conductive energy flux
`-(kappa_molecular + kappa_art) * grad_T_ion[dim]` (isothermal), using
the transport model's local conductivity.

Every species has zero total normal flux, including molecular and artificial
diffusion. The artificial `:bulk` channel's normal species and energy transport
is suppressed at the wall; isothermal heat exchange uses the conductivity above.
Pressure and viscous momentum fluxes remain, providing wall traction. The flux
condition is applied after complete assembly and before exchange/divergence,
including at corners. Filtering and hard state enforcement are separate
operations; neither guarantees a whole-domain discrete conservation identity.
"""
struct NoSlipWallBC{T<:AbstractFloat} <: BoundaryCondition
    Twall::T   # NaN → adiabatic; finite → isothermal wall
end

NoSlipWallBC(; Twall::Real=NaN) = NoSlipWallBC(float(Twall))

"""
    ExtrapolationBC()

Copy the adjacent interior state onto the physical boundary plane. This simple
zeroth-order extrapolation imposes a zero normal difference. It is not a
characteristic non-reflecting condition and may reflect outgoing disturbances.
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
    SymmetryPlaneBC()

Reflecting plane half a cell outside the first (side 1) or last (side 2) node
of the dimension it sits on. The grid on that dimension is half-offset at the
folded end, as a coordinate fold's is: a plane at the low end alone gives
h = L/(N − ½) and puts node 1 at h/2, a plane at both ends gives h = L/N and
puts node i at (i − ½)h, and no node lies on the plane itself.

Density, pressure, energy, species and the tangential velocities are even
across the plane and the normal velocity is odd. Every operator (the
derivatives, the state filter, the sensor smoother and the detector) runs its
interior stencil to the edge over the mirror halo and folds its ghost coupling
onto the matrix diagonal, so the face carries no closure row and the
discretization there is the interior one: the periodic operator on the doubled
line restricted by parity. A run between symmetry planes reproduces the
periodic run on the doubled line to round-off, and one derivative at the plane
converges at the interior order.

The parity fold carries the whole condition. `enforce!` does nothing, since no
node sits on the plane, and there is no flux correction: the fold's flux
parities already give zero normal mass, species and energy flux and zero
tangential shear traction, which is the inviscid slip wall and, under physical
viscosity, the symmetry plane's flux contract of [`SlipWallBC`](@ref).

The fold reflects one coordinate, so the metric's scale factors must not depend
on it: every dimension of `CartesianMetric` and z (dimension 3) of
`CylindricalMetric` qualify and the rest are rejected by [`validate_bc`](@ref).
The plane's dimension cannot be stretched, cannot also carry [`AxisBC`](@ref),
[`OriginBC`](@ref) or [`PoleBC`](@ref), and cannot be wrapped in a
[`SwitchableBC`](@ref). A patched or refined run does not take it;
[`SlipWallBC`](@ref) is the condition there.
"""
struct SymmetryPlaneBC <: BoundaryCondition end

"""
    InterfaceBC(neighbor)

Marker condition on a patch face that another patch supplies: a face abutting
a patch at the same level (`neighbor` is the abutting patch id), or, with
`neighbor == 0`, a fine-patch face fed by the coarse level below it (see
[`CoarseFineBC`](@ref)). There is no state to enforce and no RHS correction:
the same-level coupling runs through the interface ghost exchange, the
interface closure rows, and the shared-plane averaging (patches.jl), and a
parent-fed face's ghost layers and boundary plane are overwritten from the
prolonged coarse state after every RK stage (levels.jl); both close the line
solves with the same extended-data rows. The δ⁴ sensor detector reads those
ghost layers as well, for the fields recovered over the padded extent
(`delta4_sum!`). `Solver` substitutes this onto such faces itself; it is not a
user-supplied condition.

One type serves both faces on purpose. A patch's face conditions are part of
its `Patch` type and the right-hand side compiles once per distinct patch
type (1.8 s on the CPU backend, 3.9 s on the device backend), so a tiled
level whose faces carried two marker types would compile its drivers once
per face pattern, up to 64 in a three-dimensional nest; with one type they
compile once per level.
"""
struct InterfaceBC <: BoundaryCondition
    neighbor::Int
end

enforce!(::InterfaceBC, Q, solver, d, side) = nothing

"""
    CoarseFineBC()

The marker condition of a fine-patch face abutting the coarse level below it:
an [`InterfaceBC`](@ref) with `neighbor == 0`, so every face of a refined
patch carries the one type. `Solver` substitutes this onto refined-patch
faces itself; `parent_fed` tells the two faces apart.
"""
CoarseFineBC() = InterfaceBC(0)

"Whether a refined patch's face condition is the parent-fed one
([`CoarseFineBC`](@ref)) rather than a same-level interface."
parent_fed(bc::InterfaceBC) = bc.neighbor == 0
parent_fed(::BoundaryCondition) = false

"""
    isperiodic(bc) -> Bool

Whether `bc` joins a dimension periodically. The default for a
[`BoundaryCondition`](@ref) is `false`; custom periodic conditions must extend
this method because setup uses the answer when constructing MPI topology and
directional plans.
"""
isperiodic(::BoundaryCondition) = false
isperiodic(::PeriodicBC) = true

"""
    SwitchableBC(before, after; at = nothing)

One face that behaves as `before` until it switches, then as `after`. This
supports calculations that require one boundary condition during an interaction
and another afterwards: a wall that becomes an outflow once the waves of interest
have formed, or an inflow that injects one flow and then fires a shock.

With `at = t`, the face switches by itself at time `t`: [`run!`](@ref) ends a
step exactly at `t`, as it does for a scheduled [`AtTime`](@ref) callback, and
switches between that step and the next, so every Runge–Kutta stage of a step
sees the same condition. A [`StepControl`](@ref) rollback to before `t` restores
`before`. Without `at`, the face switches when [`switch!`](@ref) is called on
it, typically from a [`Callback`](@ref).

The wrapper carries both conditions because `bcs` sits on an immutable
[`Patch`](@ref) as an immutable tuple; a switch mutates the wrapper, not the
solver's face list.

Every rank must switch on the same step. `after` may perform collectives that
`before` does not; `NSCBCOutflowBC` is one example. Rank disagreement therefore
causes a collective-ordering deadlock. Use a [`Callback`](@ref), whose trigger
verdict is globally consistent, not a rank-local test.

Both conditions must agree on periodicity because `setup` uses that property to
construct the decomposition and line plans. Fold conditions (`AxisBC`,
`OriginBC`, `PoleBC`, and `SymmetryPlaneBC`) cannot be wrapped because `setup`
identifies them by the boundary-condition type itself and builds the grid
spacing and the folded operators from the answer.
"""
mutable struct SwitchableBC{B1<:BoundaryCondition,B2<:BoundaryCondition} <: BoundaryCondition
    before::B1
    after::B2
    switched::Bool
    at::Float64         # scheduled switch time; NaN when switched by hand
end

_is_fold_bc(bc) = bc isa AxisBC || bc isa OriginBC || bc isa PoleBC ||
                  bc isa SymmetryPlaneBC

function SwitchableBC(before::BoundaryCondition, after::BoundaryCondition;
                      at::Union{Nothing,Real}=nothing)
    isperiodic(before) == isperiodic(after) ||
        throw(ArgumentError("SwitchableBC: both conditions must agree on periodicity"))
    (_is_fold_bc(before) || _is_fold_bc(after)) &&
        throw(ArgumentError("SwitchableBC cannot wrap a fold condition " *
                            "(AxisBC, OriginBC, PoleBC, SymmetryPlaneBC); " *
                            "setup detects those by type"))
    return SwitchableBC(before, after, false, at === nothing ? NaN : Float64(at))
end

# --- Scheduled switches. `t` advances identically on every rank, so a switch
# decided from it needs no reduction, as for `AtTime`.

# Every SwitchableBC reachable from a face list, including one nested as the
# `before` or `after` of another.
_switchables!(out, bc::SwitchableBC) =
    (push!(out, bc); _switchables!(out, bc.before); _switchables!(out, bc.after); out)
_switchables!(out, bc) = out

function _switchables(bcs)
    out = SwitchableBC[]
    for d in 1:3, bc in bcs[d]
        _switchables!(out, bc)
    end
    return unique!(out)     # a condition shared by two faces appears once
end

_scheduled(bc::SwitchableBC) = !isnan(bc.at)

"The earliest scheduled switch still ahead, or `Inf`."
function next_switch_time(bcs)
    t = Inf
    for bc in _switchables(bcs)
        _scheduled(bc) && !bc.switched && (t = min(t, bc.at))
    end
    return t
end

"Switch every scheduled face whose time `t` has reached; `tol` absorbs rounding."
function apply_scheduled_switches!(bcs, t, tol)
    for bc in _switchables(bcs)
        _scheduled(bc) && !bc.switched && t >= bc.at - tol && switch!(bc)
    end
    return bcs
end

"Set every scheduled face to the side of its switch time that `t` is on."
function rewind_scheduled_switches!(bcs, t, tol)
    for bc in _switchables(bcs)
        _scheduled(bc) && (bc.switched = t >= bc.at - tol)
    end
    return bcs
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
enforce!(::SymmetryPlaneBC, Q, solver, d, side) = nothing

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
    correct_flux!(bc, solver, Q, dim, side)

Physical flux boundary hook. `compute_rhs!` calls it after complete flux assembly
(including the artificial `:bulk` channel), before flux halo exchange and compact
divergence. Mutate the physical normal flux `solver.flux[dim, component]` on the
owned wall plane. Metric area factors are applied afterwards. The default is a
no-op. `Q` has already had its hard boundary conditions enforced by the caller.

Every rank calls the hook for both faces of each active dimension, even when
`wallplane` returns `nothing`. CompactLES' implementations perform no collective
operations and return immediately on nonowners. A custom hook that communicates
must preserve the same collective order on every rank, including nonowners.
Correcting the flux here lets the compact solve carry it into all affected RHS
rows; overwriting only the endpoint RHS does not impose this condition.
"""
correct_flux!(::BoundaryCondition, solver, Q, dim, side) = nothing

correct_flux!(bc::SwitchableBC, solver, Q, dim, side) =
    bc.switched ? correct_flux!(bc.after, solver, Q, dim, side) :
                  correct_flux!(bc.before, solver, Q, dim, side)

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

# Geometry restriction of a symmetry plane: the fold reflects one coordinate
# about the plane, so no scale factor may depend on that coordinate. Cartesian
# scale factors depend on nothing, and the cylindrical (1, r, 1) do not depend
# on z. The cylindrical radius and azimuth and every spherical dimension fail
# the test, and their singular ends are already AxisBC, OriginBC and PoleBC.
function validate_bc(::SymmetryPlaneBC, metric, eos, d::Int, side::Int)
    (metric isa CartesianMetric || (metric isa CylindricalMetric && d == 3)) ||
        error("SymmetryPlaneBC on dimension $d requires CartesianMetric or " *
              "the z dimension of CylindricalMetric; on $(typeof(metric)) " *
              "dimension $d the scale factors depend on the folded coordinate")
    return nothing
end

"""
    sensor_mirror(bc) -> Bool

Whether a closed face reflects the fields the artificial-property sensors are
built from. The fourth-difference detector reads two taps past the boundary;
where this answers `true` it takes them from the node-centred mirror of the
interior, with the sign of the field across the wall. Where it answers
`false` it clamps the index instead, except at an [`InterfaceBC`](@ref) face,
whose ghost layers hold the abutting patch's or the coarse level's own values
for a field recovered over the padded extent (`delta4_sum!`).

The compact sensor operators read the same face. Where this answers `true`,
the `:gaussian` smoother and the `:d8` detector close the face with the
node-centred rows of [`wall_closures`](@ref) in place of their own half-offset
ones, the detector with one row set per sign of the field. Those rows are
fixed when the solver is built, so a face whose condition can change mid-run
takes them only when both of its conditions answer `true` here; see
`ring_sum!`.

The default is `false`. A mirror is the statement that the solution continues
past the face as its own reflection, which holds at an impermeable wall and at
no other condition here: an inflow, a Dirichlet face and a characteristic
outflow admit an arbitrary continuation, an interface end reads the data the
abutting patch or the coarse level supplied, and a fold end takes its own
half-offset mirror inside the same routine. The clamp makes no such statement.
Extend this method for a custom reflecting wall.

A [`SwitchableBC`](@ref) answers for whichever condition is active, so a face
that leaves the wall state during a run leaves the mirror with it. Every rank
switches on the same step, and the detector communicates nothing on this path,
so the two ends of a line cannot disagree.
"""
sensor_mirror(::BoundaryCondition) = false
sensor_mirror(::SlipWallBC) = true
sensor_mirror(::NoSlipWallBC) = true

sensor_mirror(bc::SwitchableBC) =
    bc.switched ? sensor_mirror(bc.after) : sensor_mirror(bc.before)

# Whether the closure rows planned at setup treat a face as a reflecting
# mirror. A plan is fixed for the run while `sensor_mirror` follows a
# `SwitchableBC`'s active condition, so a switchable face qualifies only when
# both of its conditions are mirrors; otherwise it keeps the scheme's own rows
# and the compact sensor operators fold onto the half-offset mirror there. The
# `:delta4` detector queries `sensor_mirror` per call and is unaffected.
planned_sensor_mirror(bc) = sensor_mirror(bc) === true
planned_sensor_mirror(bc::SwitchableBC) =
    planned_sensor_mirror(bc.before) && planned_sensor_mirror(bc.after)

"""
    wall_internal_energy(eos, Q, I, n_species, T_wall)

Return the internal-energy density `ρe` imposed at padded state index `I` by
an isothermal [`NoSlipWallBC`](@ref). A custom [`EOS`](@ref) must implement this
hook if it supports isothermal walls. `Q` supplies the local partial densities,
`n_species` is the equation set's species count, and `T_wall` is expressed in
the temperature units of the EOS.

The method is evaluated inside a specialized wall-plane loop and must not
perform communication or mutate `Q`.
"""
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
# per-point writes are independent, so the launch reproduces a serial plane
# loop bitwise on the host path too.

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

@inline function _slip_flux_point!(flux, d, m1, m2, m3, n_species, i_energy,
                                   o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        for sp in 1:n_species
            flux[d, sp][I] = 0
        end
        # A slip wall is a symmetry plane: the tangential velocity, density,
        # temperature and mass fractions are even about it and the normal
        # velocity is odd. The even fields' normal derivatives therefore
        # vanish, leaving no shear traction and no conduction, and the normal
        # velocity itself vanishes, so no mass, tangential momentum or energy
        # crosses the plane. Rebuild all three to zero rather than subtracting
        # terms: this also removes the bulk component flux from them, and is
        # independent of the EOS energy gauge. The normal momentum flux is
        # untouched, since pressure, the normal viscous stress and the
        # dilatational term are all even, and they carry the wall's traction.
        d == 1 || (flux[d, m1][I] = 0)
        d == 2 || (flux[d, m2][I] = 0)
        d == 3 || (flux[d, m3][I] = 0)
        flux[d, i_energy][I] = 0
    end
    return nothing
end

function correct_flux!(::SlipWallBC, solver, Q, d, side)
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    m1, m2, m3 = solver.equations.i_mom
    plane_pointwise!(_slip_flux_point!, solver.rho, plane,
                     solver.field_tuples.flux, d, m1, m2, m3,
                     solver.equations.n_species, solver.equations.i_energy)
    return nothing
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
    # The keyword constructor stores a Float64 `Twall`; convert it to the
    # state's type so an isothermal wall does not promote the energy update.
    plane_pointwise!(_no_slip_wall_point!, Q, plane, Q, solver.eos,
                     convert(eltype(Q), bc.Twall),
                     !isnan(bc.Twall), m1, m2, m3, solver.equations.i_energy,
                     solver.equations.n_species)
    nothing
end

@inline function _no_slip_flux_point!(flux, cp_mix, kappa_art, grad_T,
                                      transport, eos, T_ion, rho, Y, iso, d, n_species, i_energy,
                                      o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        for sp in 1:n_species
            flux[d, sp][I] = 0
        end
        # At a stationary noncatalytic wall neither enthalpy diffusion nor
        # convective/viscous work transports energy. Rebuild the allowed heat
        # flux rather than subtracting terms: this also removes the complete
        # bulk component flux, and is independent of the EOS energy gauge.
        if iso
            molecular = transport_at(transport, eos, T_ion, rho, cp_mix, Y, I)
            flux[d, i_energy][I] = -(molecular.kappa + kappa_art[I]) * grad_T[I]
        else
            flux[d, i_energy][I] = zero(eltype(cp_mix))
        end
    end
    return nothing
end

function correct_flux!(bc::NoSlipWallBC, solver, Q, d, side)
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    plane_pointwise!(_no_slip_flux_point!, solver.rho, plane,
                     solver.field_tuples.flux, solver.cp_mix, solver.kappa_art,
                     solver.grad_T_ion[d], solver.transport, solver.eos,
                     solver.T_ion, solver.rho, solver.field_tuples.Y,
                     !isnan(bc.Twall), d, solver.equations.n_species,
                     solver.equations.i_energy)
    return nothing
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
    # Copying the adjacent interior plane gives zeroth-order extrapolation at
    # the edge. `NSCBCOutflowBC` instead applies a characteristic correction
    # when reflected disturbances affect the solution.
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    sgn = side == 1 ? 1 : -1
    s = ntuple(k -> k == d ? sgn : 0, 3)
    plane_pointwise!(_extrapolation_point!, Q, plane, Q, s[1], s[2], s[3],
                     solver.equations.n_cons)
    nothing
end

"""
    CompositeBC(members, selector)

One face divided among several conditions. `members` is a tuple of boundary
conditions and `selector(x1, x2, x3)` returns the index of the member that
holds the face point at those coordinates. The selector is evaluated once per
face point, when a rank first applies the face, and the result is stored as a
mask over that rank's part of the face plane.

Each hook runs every member in order on every rank, so a member's collectives
are reached on ranks whose part of the face holds none of its points. Every
member acts on the whole plane from the same input, and the composite keeps
each member's result only at the points the mask assigns to it: the
boundary-plane state in [`enforce!`](@ref), the normal flux in
[`correct_flux!`](@ref) and the right-hand side in [`correct_rhs!`](@ref). A
member's writes anywhere else are not masked. The cost is two plane copies
per member per hook.

The change from one member to the next is sharp. An inflow member whose target
velocity falls to zero at the edge of its region avoids a velocity jump there.
An [`NSCBCInflowBC`](@ref) member with a pointwise `target` evaluates it over
the whole face.

The face uses the scheme's closure rows, as every nonperiodic face does.
[`sensor_mirror`](@ref) is `true` only when it is `true` for every member, so
a face mixing a wall with an open condition clamps the detector taps over the
whole face.

A member cannot be periodic, a fold, an interface marker, a
[`SwitchableBC`](@ref) or another composite. A `SwitchableBC` may wrap a
composite face.
"""
struct CompositeBC{M<:Tuple,F} <: BoundaryCondition
    members::M
    selector::F
    # (decomp, region, d, side) => _FaceScratch, filled on first use per face
    scratch::IdDict{Any,Any}
end

function CompositeBC(members::Tuple, selector)
    isempty(members) && throw(ArgumentError("CompositeBC needs at least one member"))
    for m in members
        m isa BoundaryCondition ||
            throw(ArgumentError("CompositeBC members must be BoundaryCondition objects"))
        (isperiodic(m) || _is_fold_bc(m) || m isa InterfaceBC || m isa SwitchableBC ||
         m isa CompositeBC) &&
            throw(ArgumentError("CompositeBC cannot hold a $(nameof(typeof(m))): " *
                                "periodic, fold, interface, switchable and " *
                                "composite conditions act on the whole face"))
    end
    return CompositeBC(members, selector, IdDict{Any,Any}())
end

# One face's mask and blend buffers on one rank, over the padded index box of
# its `wallplane`. `mask` and the buffers share the storage of the patch's
# fields, so the blend bodies launch on the same backend as the members.
struct _FaceScratch{M,B}
    mask::M               # member index per plane point
    present::Vector{Bool} # whether this rank's plane holds any point of member m
    saved::B              # the plane before a member acted, per component
    picked::B             # the blended result, per component
end

function _face_scratch(bc::CompositeBC, solver, plane::WallPlane, d::Int, side::Int)
    key = (solver.decomp, solver.region, d, side)
    s = get(bc.scratch, key, nothing)
    s === nothing || return s
    r1, r2, r3 = plane.indices
    dims = (length(r1), length(r2), length(r3))
    host = Array{Int32}(undef, dims)
    nm = length(bc.members)
    present = fill(false, nm)
    for (c, I) in zip(CartesianIndices(dims), plane)
        i, j, k = interior_index(solver, I)
        m = bc.selector(xcoord(solver, 1, i), xcoord(solver, 2, j),
                        xcoord(solver, 3, k))
        (m isa Integer && 1 <= m <= nm) ||
            error("CompositeBC selector returned $m at a face point of dimension " *
                  "$d side $side; expected an integer in 1:$nm")
        host[c] = m
        present[m] = true
    end
    T = eltype(solver.rho)
    mask = similar(solver.rho, Int32, dims)
    copyto!(mask, host)
    n_cons = solver.equations.n_cons
    saved = similar(solver.rho, T, (dims..., n_cons))
    picked = similar(solver.rho, T, (dims..., n_cons))
    s = _FaceScratch(mask, present, saved, picked)
    bc.scratch[key] = s
    return s
end

# A blended plane is a component of the conserved state (`Q`, `dQ`), indexed
# `A[I, c]`, or of the normal flux, indexed `A[d, c][I]`.
@inline _face_get(A, d, I, c) = @inbounds A[I, c]
@inline _face_get(A::Union{FieldMatrix,DeviceFieldMatrix}, d, I, c) = @inbounds A[d, c][I]
@inline _face_set!(A, v, d, I, c) = (@inbounds A[I, c] = v; nothing)
@inline _face_set!(A::Union{FieldMatrix,DeviceFieldMatrix}, v, d, I, c) =
    (@inbounds A[d, c][I] = v; nothing)

@inline function _face_save_point!(S, A, d, n, o1, o2, o3, i, j, k)
    I = CartesianIndex(i + o1, j + o2, k + o3)
    for c in 1:n
        @inbounds S[i, j, k, c] = _face_get(A, d, I, c)
    end
    return nothing
end

# Keep member `m`'s result where the mask selects it, then restore the saved
# plane so that the next member starts from the same input.
@inline function _face_pick_point!(R, A, S, mask, m, d, n, o1, o2, o3, i, j, k)
    I = CartesianIndex(i + o1, j + o2, k + o3)
    @inbounds mine = mask[i, j, k] == m
    for c in 1:n
        @inbounds begin
            mine && (R[i, j, k, c] = _face_get(A, d, I, c))
            _face_set!(A, S[i, j, k, c], d, I, c)
        end
    end
    return nothing
end

@inline function _face_put_point!(A, R, d, n, o1, o2, o3, i, j, k)
    I = CartesianIndex(i + o1, j + o2, k + o3)
    for c in 1:n
        @inbounds _face_set!(A, R[i, j, k, c], d, I, c)
    end
    return nothing
end

# Run `hook(member)` for every member in order, unrolled over the tuple so
# that each call is concrete.
@inline _each_member(hook::H, ::Tuple{}, m::Int) where {H} = nothing
@inline function _each_member(hook::H, members::Tuple, m::Int) where {H}
    hook(first(members), m)
    return _each_member(hook, Base.tail(members), m + 1)
end

# The shared blend of the three hooks. `A` is the array a member writes on
# the plane (`Q`, the flux collection or `dQ`), `route` the pointwise route,
# and `hook(member)` the member's own call. On a rank owning none of the
# plane every member still runs, for its collectives.
function _composite_apply!(hook::H, bc::CompositeBC, solver, A, route, d::Int,
                           side::Int) where {H}
    plane = wallplane(solver.decomp, d, side)
    if plane === nothing
        _each_member((member, m) -> hook(member), bc.members, 1)
        return nothing
    end
    s = _face_scratch(bc, solver, plane, d, side)
    n = solver.equations.n_cons
    plane_pointwise!(_face_save_point!, route, plane, s.saved, A, d, n)
    _each_member(bc.members, 1) do member, m
        hook(member)
        # A member absent from this rank's plane contributes nothing; its
        # writes are undone by the restore alone.
        plane_pointwise!(_face_pick_point!, route, plane, s.picked, A, s.saved,
                         s.mask, Int32(s.present[m] ? m : 0), d, n)
    end
    plane_pointwise!(_face_put_point!, route, plane, A, s.picked, d, n)
    return nothing
end

enforce!(bc::CompositeBC, Q, solver, d, side) =
    _composite_apply!(member -> enforce!(member, Q, solver, d, side), bc, solver,
                      Q, Q, d, side)

correct_flux!(bc::CompositeBC, solver, Q, d, side) =
    _composite_apply!(member -> correct_flux!(member, solver, Q, d, side), bc,
                      solver, solver.field_tuples.flux, solver.rho, d, side)

correct_rhs!(bc::CompositeBC, solver, Q, dQ, d, side) =
    _composite_apply!(member -> correct_rhs!(member, solver, Q, dQ, d, side), bc,
                      solver, dQ, dQ, d, side)

validate_bc(bc::CompositeBC, metric, eos, d::Int, side::Int) =
    foreach(member -> validate_bc(member, metric, eos, d, side), bc.members)

sensor_mirror(bc::CompositeBC) = all(member -> sensor_mirror(member) === true,
                                     bc.members)
planned_sensor_mirror(bc::CompositeBC) = all(planned_sensor_mirror, bc.members)

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
