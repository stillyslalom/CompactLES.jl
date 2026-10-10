# The conservative multicomponent Navier–Stokes RHS in
# orthogonal curvilinear coordinates, with collapsed (1-D/2-D) dimensions and
# a regularized cylindrical axis.
#
# Collapsed dimensions (n_global[d] == 1) carry no derivatives, filters, halos,
# or exchanges, but keep their velocity component and all metric source
# terms, as axisymmetric (r, z) flow with optional swirl
# requires. Coordinate singularities (cylindrical axis, spherical origin and
# poles, axisymmetric or fully resolved) are regularized
# by a half-offset grid, r_i = (i − ½)h, plus parity conditions: halos below
# the axis are mirror-filled with per-field signs (even scalars/u_z, odd
# u_r/u_θ), interior stencils run all the way to the first node, and the
# implicit LHS coupling to the ghost unknown is folded analytically onto the
# diagonal (per solution parity σg: derivatives flip field parity, filters
# preserve it). No node sits at r = 0 and no scale factor vanishes.

# Whether the flux through an interface end carries a molecular part through
# ghost fluxes: `interface_flux = :ghost` with transport that is not zero.
_ghost_viscous(interface_flux::Symbol, transport) =
    interface_flux === :ghost && !_zero_molecular_diffusion(transport)
_ghost_viscous(solver) = _ghost_viscous(solver.interface_flux, solver.transport)

# Under `interface_flux = :ghost` with the artificial properties on, an
# interface dimension carries the remainder F − f in `ghost_flux` and
# differences the whole flux through the gradient plans in one solve; the
# level's records fill its ghost layers at a same-level end. The divergence
# plans' one-sided rows, a second solve, took the remainder before, at an
# interface order near 3 against 6 here. Test/bench toggle, read at
# construction (allocation) and at every right-hand side; `false` restores
# the second solve.
const GHOST_FLUX_REMAINDER = Ref(true)

# With `GHOST_FLUX_REMAINDER` on, the dimensions with a coarse-fine end too,
# the remainder's ghost layers there extrapolated from the interior by a
# polynomial of degree `GHOST_REMAINDER_DEGREE[]` along the line (lower on a
# rank holding fewer nodes). Test/bench toggle; `false` leaves those
# dimensions to the second solve.
const GHOST_REMAINDER_EXTRAPOLATE = Ref(true)
const GHOST_REMAINDER_DEGREE = Ref(5)

_ghost_remainder(interface_flux::Symbol, art) =
    GHOST_FLUX_REMAINDER[] && interface_flux === :ghost && art.enabled
_ghost_remainder(solver) = _ghost_remainder(solver.interface_flux, solver.art)

# --- Operator routing through folds ----------------------------------------

# The plans tuple is heterogeneous when a collapsed or folded dimension puts
# `nothing` in one of its slots. Indexing that tuple with a runtime `d` yields a
# union that the caller must split.
#
# The measured allocation was about 330 B per operator application and 11.9 kB
# per RHS on a planar (32, 16, 1) run, compared with 336 B per RHS in 3-D.
# Branching on `d` (`_plan_at`) reduces this to 160 B per application and
# 3.8 kB per RHS: the plan the branches merge is a value of the union of its
# type and `Nothing`, and such a value is boxed. `_operator_plan` removes that
# box as well: a slot holding `nothing` raises there, so the branches merge
# only plans, which share one type, and the operator routing below goes
# through it.
#
# A concrete sentinel plan would remove the union at its source, but
# constructing one requires a `LineSolver` and communicator for a dimension
# that is never swept. The explicit branch avoids that unused state.
#
# The tuple's shape stays in the `Patch` type for the same reason, and this is
# the one place where taking configuration out of that type does not pay. Two
# shape-independent storages were tried and both were rejected on
# `bench/audit.jl`, against a baseline of 16 B per `compute_rhs!` at 48³:
# `NTuple{3,Union{Nothing,AbstractDirPlan}}`, which makes every `apply_along!`
# a dynamic dispatch, measured 18.3 kB, and `NTuple{3,Union{Nothing,PL}}` at a
# concrete plan kind `PL`, which keeps the call static and still measured
# 5968 B — the union reaches the tuple field, not just the call. Either
# collapses the workload's `Patch` types from 16 to 11 and neither is worth
# that on the flagship path. Tuple covariance also defeats the second one at
# construction: converting a tuple to a wider tuple type returns the narrow
# value, so an all-`nothing` tuple leaves `PL` unconstrained.
@inline _plan_at(plans::Tuple, d::Int) = d == 1 ? plans[1] : d == 2 ? plans[2] : plans[3]

# The plan of an operator applied along `d`. A slot holding `nothing` raises,
# as an operator applied along a dimension without a plan would, so that
# branch yields no value and the result is inferred as the type of the slots
# holding a plan, not as its union with `Nothing`.
@inline _operator_plan(plans::Tuple, d::Int) =
    d == 1 ? _some_plan(plans[1]) : d == 2 ? _some_plan(plans[2]) : _some_plan(plans[3])
@inline _some_plan(plan) = plan
_some_plan(::Nothing) = throw(ArgumentError("no operator plan along this dimension"))

# Whether the plan along `d` is a device plan, tested per tuple position for the
# same reason.
@inline _device_plan_at(plans::Tuple, d::Int) =
    d == 1 ? plans[1] isa DevicePlan :
    d == 2 ? plans[2] isa DevicePlan : plans[3] isa DevicePlan

# The fold on dimension `d`, or `nothing`. A tuple holding a fold beside
# `nothing` is heterogeneous, and indexing it with a runtime `d` boxes the
# tuple; at a constant position the fold, a mutable object, is read as the
# reference it is.
@inline _fold_at(solver, d::Int) = _plan_at(solver.folds, d)

# The operator routers below are `@noinline`. Each is small enough that Julia
# would inline it into every caller, which then carries its fold and mask
# branches; called, a router is compiled once per argument types, and the
# package image holds the instances its workload reaches.

"""
    deriv_along!(out, f, solver, d, σf)

Compact derivative of `f` along active dimension `d`; `σf` is the field's
antipodal sign for the fold on `d` (ignored when there is none). The caller
ensures current rank-boundary halos of `f`. Only the interior of `out` is
written.

This is a distributed line solve along `d`, so it is collective over that
dimension's sub-communicator and every rank must call it, including ranks
holding no part of a fold. At a self-paired fold `f` is also written:
`fold_fill!` mirrors its halos beyond the folded end before the sweep. A paired
fold leaves `f` untouched, running the even/odd butterfly through
`solver.pairbuf` instead.
"""
@noinline function deriv_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = _fold_at(solver, d)
    if fold === nothing
        apply_along!(out, _operator_plan(solver.deriv_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:deriv))
    end
    return _mask_child_derivative!(out, f, solver, d, σf, Val(:deriv), false)
end

"""
    div_along!(out, f, solver, d, σf)

Compact derivative of `f` along `d` through the divergence plans. These are
`solver.deriv_plans` except at a patch-interface end under
`interface_rhs = :extended`, where the gradient plans read exchanged ghost
data that a flux array does not carry, so the divergence keeps one-sided
closure rows there, the scheme's own or the cascade's for the neutral set
(`interface_divergence_closures`, `solver.div_plans`). A folded dimension draws on the
fold's own divergence plans instead, which are its derivative plans except on a
refined patch whose folded dimension ends at an interface. The flux-divergence
loop and the discrete-GCL construction `gcl_cotr!` go through here so the two
apply the identical operator. Same collective, halo, and fold contract as
[`deriv_along!`](@ref).
"""
@noinline function div_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = _fold_at(solver, d)
    if fold === nothing
        apply_along!(out, _operator_plan(solver.div_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:div))
    end
    return out
end

# Whether a derivative along the folded dimension `d` takes the fused scatter
# of an unfolded one: a self-paired fold, whose route is the mirror fill and
# one host solve through the fold's plan, on a patch with no derivative mask
# along `d`. The fused scatter applies the same product or subtraction to the
# same solve output as the separate pass, so the interior is bitwise the same,
# and it saves that pass over the whole array.
@inline _fused_fold(solver, ::Nothing, d::Int) = false
@inline _fused_fold(solver, fold::FoldSpec, d::Int) =
    fold.pair === nothing && !(fold_dplan(fold, 1) isa DevicePlan) &&
    !_child_masked(solver, d)

"""
    deriv_scaled_along!(out, f, solver, d, σf)

Compact derivative of `f` along active dimension `d`, scaled pointwise by
`solver.inv_h[d]` inside the scatter of the line solve. Interior results are
bit-identical to [`deriv_along!`](@ref) followed by `_scale_grad!`; only halo
cells of `out` differ (the two-pass rescale also scaled them, but they hold no
data any consumer reads without a fresh exchange). A self-paired fold takes
the same fused scatter after its mirror fill; a paired fold, a device plan
and a parent patch whose derivatives along `d` take the child mask take the
two-pass route unchanged. Same collective, halo, and fold contract as
`deriv_along!`.
"""
@noinline function deriv_scaled_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = _fold_at(solver, d)
    if fold === nothing && !_device_plan_at(solver.deriv_plans, d)
        apply_along_scaled!(out, _operator_plan(solver.deriv_plans, d), f, solver.decomp,
                            solver.inv_h[d])
        _mask_child_derivative!(out, f, solver, d, σf, Val(:deriv), true)
    elseif _fused_fold(solver, fold, d)
        # A self-paired fold is the mirror fill and the plain solve through
        # the fold's plan, so the scale rides the scatter as above. A masked
        # parent keeps the two-pass route, whose mask is subtracted before
        # the scale.
        fold_fill!(f, solver.decomp, d, fold.lo, fold.hi, σf)
        apply_along_scaled!(out, _fold_plan(fold, σf, Val(:deriv), 1, false), f,
                            solver.decomp, solver.inv_h[d])
    else
        deriv_along!(out, f, solver, d, σf)
        _scale_grad!(out, solver, d)
    end
    return out
end

"""
    div_subtract_along!(dQ, c, f, solver, d, σf, inv_J)

Compact derivative of `f` along `d` through the divergence plans, subtracted
from conserved component `c` of `dQ` inside the scatter of the line solve:
`dQ[I, c] -= inv_J[I] * (D f)[I]`, or `dQ[I, c] -= (D f)[I]` when
`inv_J === nothing` (the unit-geometry case). Interior results are
bit-identical to [`div_along!`](@ref) into scratch followed by the
subtraction pass. A self-paired fold takes the same fused scatter after its
mirror fill; a paired fold, a device plan and a parent patch whose
derivatives along `d` take the child mask take the two-pass route through
`solver.tmp_a`. Same collective, halo, and fold contract as `div_along!`.
"""
@noinline function div_subtract_along!(dQ, c::Int, f, solver::SolverLike, d::Int,
                             σf::Int, inv_J)
    fold = _fold_at(solver, d)
    _reflux_open!(solver, dQ, c, d, inv_J)
    if fold === nothing && !_device_plan_at(solver.div_plans, d)
        apply_along_subtract!(dQ, c, _operator_plan(solver.div_plans, d), f, solver.decomp,
                              inv_J)
    elseif _fused_fold(solver, fold, d)
        fold_fill!(f, solver.decomp, d, fold.lo, fold.hi, σf)
        apply_along_subtract!(dQ, c, _fold_plan(fold, σf, Val(:div), 1, false), f,
                              solver.decomp, inv_J)
    else
        div_along!(solver.tmp_a, f, solver, d, σf)
        nx, ny, nz = solver.decomp.n_local
        o1, o2, o3 = solver.decomp.n_halo_d
        if inv_J === nothing
            pointwise!(_subtract_div_point!, dQ, nx, ny, nz,
                       dQ, solver.tmp_a, c, o1, o2, o3)
        else
            pointwise!(_subtract_jac_div_point!, dQ, nx, ny, nz,
                       dQ, solver.tmp_a, inv_J, c, o1, o2, o3)
        end
    end
    _mask_child_divergence!(dQ, c, f, solver, d, σf, inv_J, Val(:div))
    _reflux_close!(solver, dQ, c, f, d, inv_J)
    return dQ
end

# The pressure term inv_h_d ∂p/∂ξ_d of component `c` where it enters as a
# gradient (`_pressure_gradient`), subtracted from `dQ`. A separate function so
# that the positivity limiter's stage array keeps it in a register of its own.
pressure_subtract_along!(dQ, c::Int, solver::SolverLike, d::Int) =
    div_subtract_along!(dQ, c, solver.p, solver, d, 1, solver.inv_h[d])

# --- Derivatives on a parent level ---------------------------------------------
#
# A compact derivative is A⁻¹ B f: the explicit stencil B f is local, and A⁻¹
# spreads it along the line, decaying by about 0.38 per node for the sixth-order
# tridiagonal scheme. On a parent patch the nodes a child covers carry the
# child's restricted solution, and a shock there is a jump narrower than the
# parent spacing. The parent's gradients and flux divergences then carry an
# alternating tail of that jump to the nodes it evolves itself: the uncovered
# ones and the covered margin beside a coarse-fine face, which restriction does
# not overwrite and from which the child's shell is interpolated. The child took
# the tail in through its face data and, at a regrid, through the fill of the
# nodes the moving level newly covers, so a level a few parent nodes ahead of a
# converging shock carried tens of times the uniform fine grid's disturbance
# ahead of it. Neither the filter (which `_child_mask!` already masks), the
# artificial coefficients nor the regrid fill carries it; the tail of the
# gradients and the divergences together does, and either alone leaves most of
# it.
#
# Each derivative on such a patch therefore drops the explicit stencil's source
# at the nodes the parent cannot resolve, D f − A⁻¹ M B f, with M one where a
# node is deep inside a child (`child_deep`: one that restriction overwrites
# after every step, so that the parent's own values there are discarded anyway,
# and `CHILD_DERIVATIVE_DEPTH` nodes inside the child along the line), the
# density along the line is unresolved at it or within `CHILD_MASK_REACH` nodes
# (the test of `_child_mask!`), and the test reads no node beyond a
# non-periodic end of the patch's line. Every route takes it: the gradients,
# the flux divergence and the ghost-differenced divergence. A masked node's own
# derivative is then nearly zero, and the covered feature stays close to its
# restricted state through the parent's step. Applying the dropped source at
# the masked nodes alone, which keeps the line's discrete conservation, left
# five to ten times more ahead of the shock. Where no node is flagged the
# derivative is the plain one, bit for bit, and the extra solve is skipped:
# `_arm_child_mask!` decides per dimension once per right-hand side and
# reduces the decision over the patch's communicator, so the ranks of a line
# solve together or not at all. It builds the mask of every dimension once,
# from the evaluation's primitives, and every derivative of the evaluation
# reads it; where the line solve is local to the rank, the extra solve takes
# only the lines through a masked node. A paired fold's butterfly and a
# stacked level stay plain, as in the filter's mask, and so does every patch of a solver
# under the positivity limiter: its stage face fluxes are a running sum of the
# divergence, and a divergence that drops part of a jump's flux carried the
# difference to every face beyond the masked nodes, while one that restores it
# at the masked nodes alone gave the limiter an update it does not bound
# there; both left inadmissible states or a wrong solution on the limiter's
# refined Noh and Woodward–Colella rows. The resolution threshold is the
# default density tag's hold level, so under the default tagging a masked node
# lies where the density tag holds the child (`_tag_delta4_point!`) whatever a
# criterion reading the parent's derivatives or artificial coefficients finds
# there. The measurements are under bench/movinglevel.jl in
# reference/CALIBRATION_APPENDIX.md.
#
# The tail the mask removes also carries part of the mass the parent's
# uncovered nodes exchange through a coarse-fine face: with no flux
# correction at the face, a Sod shock crossing a level drifted five to twenty
# times as far with the mask, whatever the step and with the dropped source
# restored at the masked nodes. The conservative coupling (src/reflux.jl)
# corrects the junction fluxes where the parent's uncovered nodes hold an
# unresolved feature, and with it the masked crossings conserve mass to 1e-5.

# `true`, the default, gives every parent patch the masked derivatives;
# `false` gives every patch the plain ones.
const MASK_CHILD_DERIVATIVE = Ref(true)

# A node may take the mask when it and this many nodes on either side of it
# along the line are covered over their whole cells, so the first is the
# fourth node inside a face. The restriction leaves the first node inside to
# the parent, and its explicit stencil reads up to the third, so no node the
# parent evolves reads one whose evolution the mask changes. Starting a node
# deeper left far more ahead of a converging shock.
const CHILD_DERIVATIVE_DEPTH = 3

# Whether this patch's derivatives may take the mask: it carries the mask's
# scratch (a solver of more than one level), the solver has no positivity
# limiter, and a child level holds a patch. Read from the configuration and
# the level hierarchy every rank of the patch holds; a stacked level's
# workspace views carry no scratch.
_child_mask_patch(solver::SolverLike) = false
function _child_mask_patch(ps::PatchSolver)
    MASK_CHILD_DERIVATIVE[] || return false
    (isempty(ps.child_mask) || isempty(ps.child_deep)) && return false
    getfield(ps.solver, :positivity) === nothing || return false
    levels = getfield(ps.solver, :levels)
    child = ps.patch.level + 2
    return child <= length(levels) && !isempty(levels[child].transfers)
end

# Whether dimension `d` of the patch can take the mask: active, and not a
# paired fold.
function _child_mask_dim(solver::SolverLike, d::Int)
    fold = _fold_at(solver, d)
    return solver.decomp.active[d] && (fold === nothing || fold.pair === nothing)
end

"""
    _arm_child_mask!(solver)

Record on the patch, per dimension, whether any node takes the parent-level
derivative mask (`Patch.child_masked`), from the current primitives, and hold
the mask for the derivatives of the right-hand side (`ChildMaskRecord`).
Collective over the patch's communicator when the patch has a child level; a
no-op otherwise. The right-hand side calls it after its primitives are refreshed
and before its first derivative (`_gradient_step!`), and releases the mask at
its end (`_release_child_mask!`).
"""
_arm_child_mask!(solver::SolverLike) = solver
function _arm_child_mask!(ps::PatchSolver)
    _child_mask_patch(ps) || return ps
    dims = 0
    for d in 1:3
        _child_mask_dim(ps, d) || continue
        _child_mask!(ps, d, dims)
        dims |= 1 << (d - 1)
    end
    bits, lines = _child_mask_lines(ps.child_mask, ps.decomp, dims)
    record = _child_mask_record(ps)
    record.patch = ps.patch.covered
    record.lines = lines
    bits = MPI.Allreduce(bits, MPI.BOR, ps.patch.comm)
    ps.patch.child_masked = ntuple(d -> isodd(bits >> (d - 1)), 3)
    return ps
end

# The end of the evaluation that armed the mask: a later derivative of the
# patch, a diagnostic's or the next stage's, builds the mask afresh from the
# primitives it finds. Returns `out`.
_release_child_mask!(solver::SolverLike, out=solver) = out
function _release_child_mask!(ps::PatchSolver, out=ps)
    record = _child_mask_record(ps)
    record.patch === ps.patch.covered && (record.patch = SCRATCH_UNFILLED)
    return out
end

@inline _child_mask_record(ps::PatchSolver) =
    getfield(getfield(ps, :patch), :rhs_workspace).child_mask_record

@inline _child_masked(solver::SolverLike, d::Int) = false
@inline _child_masked(ps::PatchSolver, d::Int) =
    ps.patch.child_masked[d] && _child_mask_patch(ps)

# The mask along `d` and the range of the plan's lines holding its nodes: the
# mask the right-hand side armed while it holds the workspace's, or one built
# here from the current primitives, with every line.
function _child_mask_along!(ps::PatchSolver, d::Int)
    record = _child_mask_record(ps)
    held = record.patch === ps.patch.covered && record.patch !== SCRATCH_UNFILLED
    held && return ps.child_mask, record.lines[d]
    record.patch = SCRATCH_UNFILLED
    _child_mask!(ps, d, 0)
    return ps.child_mask, _all_lines(ps.decomp)[d]
end

# Every line of the plans along each dimension.
_all_lines(decomp::Decomp) = (n = decomp.n_local;
                              (1:n[2] * n[3], 1:n[1] * n[3], 1:n[1] * n[2]))

# M along `d` into bit d − 1 of `child_mask`, from the density's undivided
# fourth difference, keeping the bits of the dimensions in `keep` that an
# earlier launch wrote and clearing the others.
function _child_mask!(ps::PatchSolver, d::Int, keep::Int)
    decomp = ps.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    m = ps.child_mask
    reach = clamp(CHILD_MASK_REACH[], 0, decomp.n_halo_d[d] - 2)
    # The rows whose test reads only nodes of the line; every row on a periodic
    # line.
    n = decomp.n_global[d]
    lo, hi = decomp.periodic[d] ? (typemin(Int), typemax(Int)) :
                                  (reach + 3, n - reach - 2)
    pointwise!(_child_derivative_mask_point!, m, nx, ny, nz, m, ps.child_deep, ps.rho,
               eltype(m)(CHILD_MASK_THRESHOLD[]), reach, lo, hi, decomp.offset[d], d,
               keep, o1, o2, o3)
    return m
end

@inline function _child_derivative_mask_point!(m, deep, rho, thr, reach, lo, hi, off, d,
                                               keep, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        e = CartesianIndex(Int(d == 1), Int(d == 2), Int(d == 3))
        under = false
        for t in -reach:reach
            J = I + t * e
            δ4 = rho[J - 2e] - 4 * rho[J - e] + 6 * rho[J] - 4 * rho[J + e] + rho[J + 2e]
            under |= abs(δ4) > thr * abs(rho[J])
        end
        row = (d == 1 ? i : d == 2 ? j : k) + off
        along = isodd(unsafe_trunc(Int, deep[I]) >> (d - 1))
        drop = along & under & (lo <= row) & (row <= hi)
        kept = unsafe_trunc(Int, m[I]) & keep
        m[I] = eltype(m)(kept | (Int(drop) << (d - 1)))
    end
    return nothing
end

# The bits of the dimensions in `dims` that some interior node of `m` holds,
# and per dimension the range of the plan's lines through those nodes. Node
# (i, j, k) lies on line j + n₂(k − 1) of the first dimension's plans, which
# hold a line per column of their buffer, and on line i + n₁(k − 1) or
# i + n₁(j − 1) of the second's and the third's, which hold a line per row
# (operators.jl). Device storage reduces the bits alone and keeps every line,
# since its solve takes all of them.
function _child_mask_lines(m, decomp::Decomp, dims::Int)
    bits = mapreduce(x -> unsafe_trunc(Int, x), |, m; init=0) & dims
    return bits, _all_lines(decomp)
end
function _child_mask_lines(m::Array, decomp::Decomp, dims::Int)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    bits = 0
    lo1 = lo2 = lo3 = typemax(Int)
    hi1 = hi2 = hi3 = 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        b = unsafe_trunc(Int, m[i + o1, j + o2, k + o3]) & dims
        b == 0 && continue
        bits |= b
        if isodd(b)
            l = j + ny * (k - 1)
            lo1, hi1 = min(lo1, l), max(hi1, l)
        end
        if isodd(b >> 1)
            l = i + nx * (k - 1)
            lo2, hi2 = min(lo2, l), max(hi2, l)
        end
        if isodd(b >> 2)
            l = i + nx * (j - 1)
            lo3, hi3 = min(lo3, l), max(hi3, l)
        end
    end
    return bits, (lo1:hi1, lo2:hi2, lo3:hi3)
end

# The interior right-hand-side coefficients of a derivative plan, padded with
# zeros to the halo width so that the launch takes a fixed-length tuple.
_derivative_stencil(plan::DevicePlan) = _derivative_stencil(plan.host)
function _derivative_stencil(plan)
    ci = plan.ci
    T = eltype(ci)
    length(ci) <= 4 || error("the derivative mask takes a half-width of at most 4")
    return ntuple(q -> q <= length(ci) ? ci[q] : zero(T), 4)
end

# The plan of the derivative being corrected: the divergence or the gradient
# plans along `d`, or the fold's own with the field's sign.
function _child_mask_plan(solver, d, σf, role::Val)
    fold = _fold_at(solver, d)
    return fold === nothing ? _operator_plan(_role_plans(solver, role), d) :
                              _fold_plan(fold, σf, role, 1, false)
end
_role_plans(solver, ::Val{:div}) = solver.div_plans
_role_plans(solver, ::Val{:deriv}) = solver.deriv_plans

# sgn · A⁻¹ M B f into `child_solve`, which is returned. The mask is the one
# the right-hand side armed from its primitives (`_child_mask_along!`); `f`
# carries current halos along `d` (exchanged, or mirror-filled at a fold by
# the derivative just taken). Collective along `d`, as the derivative is.
function _child_masked_solve!(ps::PatchSolver, plan, f, d::Int, sgn::Int)
    decomp = ps.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    m, lines = _child_mask_along!(ps, d)
    s = ps.child_solve
    nlo, nhi = _rows_closed(plan)
    pointwise!(_child_source_point!, s, nx, ny, nz, s, f, m, _derivative_stencil(plan),
               d, nlo + 1, decomp.n_local[d] - nhi, sgn, o1, o2, o3)
    _solve_child_lines!(s, plan, decomp, lines)
    return s
end

# sgn times the interior row's explicit stencil of `f` at the masked nodes of
# the interior rows, zero elsewhere.
@inline function _child_source_point!(s, f, m, ci, d, row_lo, row_hi, sgn, o1, o2, o3,
                                      i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        e = CartesianIndex(Int(d == 1), Int(d == 2), Int(d == 3))
        acc = zero(eltype(s))
        for q in 1:4
            acc += ci[q] * (f[I + q * e] - f[I - q * e])
        end
        row = d == 1 ? i : d == 2 ? j : k
        masked = isodd(unsafe_trunc(Int, m[I]) >> (d - 1))
        keep = (row_lo <= row) & (row <= row_hi) & masked
        s[I] = ifelse(keep, sgn * acc, zero(acc))
    end
    return nothing
end

# A⁻¹ s in place along the plan's dimension, solving only the plan's lines in
# `lines`, outside which `s` is zero. A local solve (one rank along the line,
# no periodic wrap) eliminates each line by itself, with the same operations
# in the same order whichever lines it is given (`solve_cols!` and the
# transposed sweeps), and a zero line stays +0.0 where every pivot is
# positive, so the solve over `lines` alone matches the solve over every line
# bit for bit. The reduced interface stage of a decomposed or periodic line
# couples lines across ranks and its dense solve is not bitwise invariant to
# the number of lines (`_reduced_ldiv!`); that solve and a device plan take
# every line, so every rank of a line still enters the solve.
_solve_child_lines!(s, plan, decomp::Decomp, lines) = solve_along!(s, plan, s, decomp)
function _solve_child_lines!(s, plan::Union{DirPlan,BandPlan}, decomp::Decomp,
                             lines::UnitRange{Int})
    ls = plan.line_solver
    whole = ls.hasred || lines == 1:plan.lines || last(lines) > plan.lines ||
            !_zero_preserving(ls)
    whole && return solve_along!(s, plan, s, decomp)
    isempty(lines) && return s
    B = plan.B
    if plan.dim == 1
        _child_gather_columns!(B, s, plan.n, decomp, lines)
        solve_lines!(view(B, :, lines), ls)
        _child_scatter_columns!(s, B, plan.n, decomp, lines)
    else
        _child_gather_rows!(B, s, plan.n, decomp, plan.dim, lines)
        solve_lines_t!(view(B, lines, :), ls)
        _child_scatter_rows!(s, B, plan.n, decomp, plan.dim, lines)
    end
    return s
end

# Whether the elimination maps a zero line to +0.0 throughout: its forward
# sweep always does, and its back substitution multiplies by the reciprocal
# pivots.
_zero_preserving(ls::LineSolver) = ls.explicit || all(>(0), ls.F.dinv)
_zero_preserving(ls::BandLineSolver) = all(>(0), view(ls.F.U, 1, :))

# The lines `lines` of `s` into the columns of the first dimension's buffer
# `B` (n × lines) and back, as `_gather_lines!` and `_scatter_lines!` move
# every line.
function _child_gather_columns!(B, s, n::Int, decomp::Decomp, lines::UnitRange{Int})
    o1, o2, o3 = decomp.n_halo_d
    ny = decomp.n_local[2]
    @threaded length(lines) * n for l in lines
        kk, jj = divrem(l - 1, ny)
        @inbounds for i in 1:n
            B[i, l] = s[i + o1, jj + 1 + o2, kk + 1 + o3]
        end
    end
    return B
end

function _child_scatter_columns!(s, B, n::Int, decomp::Decomp, lines::UnitRange{Int})
    o1, o2, o3 = decomp.n_halo_d
    ny = decomp.n_local[2]
    @threaded length(lines) * n for l in lines
        kk, jj = divrem(l - 1, ny)
        @inbounds for i in 1:n
            s[i + o1, jj + 1 + o2, kk + 1 + o3] = B[i, l]
        end
    end
    return s
end

# The same for the rows of the second or third dimension's buffer
# (lines × n), whose line i + n₁(kk − 1) is the line through x index i and
# the remaining index kk (`_gather_t!`, `_scatter_t!`).
function _child_gather_rows!(B, s, n::Int, decomp::Decomp, d::Int,
                             lines::UnitRange{Int})
    nx = decomp.n_local[1]
    k_lo = (first(lines) - 1) ÷ nx + 1
    nk = (last(lines) - 1) ÷ nx + 2 - k_lo
    @threaded nk * n * nx for jk in outer_indices(n, nk)
        jr, kq = Tuple(jk)
        _child_row_block!(B, s, decomp, d, lines, jr, kq + k_lo - 1, false)
    end
    return B
end

function _child_scatter_rows!(s, B, n::Int, decomp::Decomp, d::Int,
                              lines::UnitRange{Int})
    nx = decomp.n_local[1]
    k_lo = (first(lines) - 1) ÷ nx + 1
    nk = (last(lines) - 1) ÷ nx + 2 - k_lo
    @threaded nk * n * nx for jk in outer_indices(n, nk)
        jr, kq = Tuple(jk)
        _child_row_block!(B, s, decomp, d, lines, jr, kq + k_lo - 1, true)
    end
    return s
end

# Row `jr` of the lines of `lines` through the remaining index `kk`, from `s`
# into `B` or, under `back`, from `B` into `s`.
@inline function _child_row_block!(B, s, decomp::Decomp, d::Int, lines, jr::Int,
                                   kk::Int, back::Bool)
    o1, o2, o3 = decomp.n_halo_d
    nx = decomp.n_local[1]
    base = (kk - 1) * nx
    i_lo = max(1, first(lines) - base)
    i_hi = min(nx, last(lines) - base)
    j, k = d == 2 ? (jr, kk) : (kk, jr)
    @inbounds for i in i_lo:i_hi
        if back
            s[i + o1, j + o2, k + o3] = B[base + i, jr]
        else
            B[base + i, jr] = s[i + o1, j + o2, k + o3]
        end
    end
    return nothing
end

"""
    _mask_child_derivative!(out, f, solver, d, σf, role, scaled)

On a parent patch whose last right-hand side armed dimension `d`, subtract
A⁻¹ M B f from the derivative `out` of `f` taken through the plans of `role`
(`Val(:deriv)` or `Val(:div)`), times `inv_h` under `scaled`; `out` unchanged
otherwise. Collective along `d`.
"""
@inline _mask_child_derivative!(out, f, solver::SolverLike, d::Int, σf::Int, role::Val,
                                scaled::Bool) = out
@noinline function _mask_child_derivative!(out, f, ps::PatchSolver, d::Int, σf::Int,
                                           role::Val, scaled::Bool)
    _child_masked(ps, d) || return out
    s = _child_masked_solve!(ps, _child_mask_plan(ps, d, σf, role), f, d, 1)
    scaled && _scale_grad!(s, ps, d)
    decomp = ps.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    pointwise!(_subtract_interior_point!, out, nx, ny, nz, out, s, o1, o2, o3)
    return out
end

# The same for a divergence already subtracted from component `c` of `dQ`,
# times `inv_J` where it is not `nothing`: dQ += inv_J A⁻¹ M B f.
@inline _mask_child_divergence!(dQ, c::Int, f, solver::SolverLike, d::Int, σf::Int,
                                inv_J, role::Val) = dQ
@noinline function _mask_child_divergence!(dQ, c::Int, f, ps::PatchSolver, d::Int,
                                           σf::Int, inv_J, role::Val)
    _child_masked(ps, d) || return dQ
    s = _child_masked_solve!(ps, _child_mask_plan(ps, d, σf, role), f, d, -1)
    decomp = ps.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    if inv_J === nothing
        pointwise!(_subtract_div_point!, dQ, nx, ny, nz, dQ, s, c, o1, o2, o3)
    else
        pointwise!(_subtract_jac_div_point!, dQ, nx, ny, nz, dQ, s, inv_J, c, o1, o2, o3)
    end
    return dQ
end

"""Compact filter of `f` along dimension `d` with antipodal sign `σf`.

Every rank in the directional sub-communicator must call this function. Its
halo and fold contract matches `deriv_along!`.
"""
@noinline function filt_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = _fold_at(solver, d)
    if fold === nothing
        apply_along!(out, _operator_plan(solver.filter_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:filter))
    end
    return out
end

"""
    smooth_along!(out, f, solver, d, σf)

Sensor smoother of `f` along dimension `d` with antipodal sign `σf`. This is
the Cook test filter, selected by `ArtificialProperties.smoother`, and is a distinct
operator from `filt_along!`: the two coincide only under
`ArtificialProperties(smoother = :compact)`, which aliases the filter plans and avoids
planning an operator of its own; the default `:gaussian` plans the explicit
nine-point stencil of [`gaussian_filter`](@ref). Only the artificial-property
sensors go through here, by way of `smooth!`.

A reflecting wall face is closed by the even rows of [`wall_closures`](@ref)
under `:gaussian`, since the fields smoothed here are even at a wall. The
`:compact` smoother keeps the state filter's plans and its own rows.

A refined patch holds an `InterfaceSmoothPlans` per dimension. `ghosts`
selects between its two plans: rows reading the ghost layers of `f` at every
interface face, which a tiled level's artificial-property pass fills
(`_level_artificial!`), or the closure rows a field without interface ghosts
takes. A patch of any other kind ignores it.

Every rank in the directional sub-communicator must call this function, as for
`deriv_along!`.
"""
@noinline function smooth_along!(out, f, solver::SolverLike, d::Int, σf::Int,
                       ghosts::Bool=false)
    fold = _fold_at(solver, d)
    if fold === nothing
        apply_along!(out, _smooth_plan(_operator_plan(solver.smooth_plans, d), ghosts), f,
                     solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:smooth), 1, ghosts)
    end
    return out
end

"""
    InterfaceSmoothPlans(ghost, closed)

The sensor-smoother plans of a refined patch along one dimension. `ghost`
closes an interface end, same-level or coarse-fine, with the smoother's
interface rows, which read the ghost layers; `closed` keeps the rows of a
field whose interface ghosts carry no data, the scheme's own. A face on the
domain boundary takes the same rows in both. `smooth_along!` selects between
them.
"""
struct InterfaceSmoothPlans{P}
    ghost::P
    closed::P
end

@inline _smooth_plan(plan, ghosts::Bool) = plan
@inline _smooth_plan(plans::InterfaceSmoothPlans, ghosts::Bool) =
    ghosts ? plans.ghost : plans.closed

"""
    ring_along!(out, f, solver, d, σf, σw = 1, ghosts = false)

Compact eighth derivative of `f` along dimension `d` with antipodal sign `σf`,
the ringing detector selected by `ArtificialProperties(detector = :d8)`. Only `ring_sum!`
calls this, and only under that setting: `solver.ring_plans` is `nothing`
otherwise, which keeps this function off the default configuration's inference
path. Indexing that field under `:delta4` would throw. See `detect_sum!`.

`σw` is the field's sign across a reflecting wall on this dimension, and picks
the plan whose closure rows fold onto the node-centred mirror with that sign
([`wall_closures`](@ref)). Each dimension carries the two plans as a pair, so
the choice is a tuple index rather than a branch. Where neither face is such a
wall the pair holds one plan twice and the index is immaterial.

A refined patch holds an `InterfaceRingPlans` per dimension instead.
`ghosts` selects between its two plan pairs: rows reading the interface ghost
layers of `f`, or the scheme's own closure rows, which read none; `σw` then
picks the wall sign within the pair, as for the root, where the patch reaches
a reflecting face of the domain.

Every rank in the directional sub-communicator must call this function. Its
halo and fold contract matches `deriv_along!`.
"""
@noinline function ring_along!(out, f, solver::SolverLike, d::Int, σf::Int, σw::Int=1,
                     ghosts::Bool=false)
    fold = _fold_at(solver, d)
    if fold === nothing
        apply_along!(out, _ring_plan(_operator_plan(solver.ring_plans, d), σw, ghosts), f,
                     solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:ring), σw, ghosts)
    end
    return out
end

# The wall-sign half of a ring plan pair. The two plans differ only in their
# closure rows, so the selection is a tuple index and adds nothing to the hot
# path; see the comment above `_plan_at` for why the dimension is indexed the
# same way.
@inline _wall_at(pair, σw::Int) = σw > 0 ? pair[1] : pair[2]

"""
    InterfaceRingPlans(ghost, closed)

The `detector = :d8` plans of a refined patch along one dimension. A face of
a refined patch is a coarse-fine or same-level interface end, or a face on
the domain boundary carrying the root's condition. `ghost` closes an
interface end with rows that read the ghost layers, for a field recovered
over the padded extent; `closed` keeps the scheme's own closure rows, for a
field whose interface ghosts carry no data (the strain magnitude, the
dilatation). Each is a pair indexed by the field's sign across a reflecting
face on the domain boundary, whose wall rows both take, aliased to one plan
where the dimension has no such face. `ring_along!` selects between them.
"""
struct InterfaceRingPlans{P}
    ghost::Tuple{P,P}
    closed::Tuple{P,P}
end

@inline _ring_plan(pair::Tuple, σw::Int, ghosts::Bool) = _wall_at(pair, σw)
@inline _ring_plan(plans::InterfaceRingPlans, σw::Int, ghosts::Bool) =
    _wall_at(ghosts ? plans.ghost : plans.closed, σw)

# Scale a raw coordinate-derivative field by 1/h_d pointwise (full array).
@inline function _scale_grad_point!(g, ih, i, j, k)
    @inbounds g[i, j, k] *= ih[i, j, k]
    return nothing
end

function _scale_grad!(g, solver, d)
    ih = solver.inv_h[d]
    # The patch's padded extent, not `size(g)`: on a stacked level `g` spans
    # every tile and the launch runs the one-tile box per tile.
    n1, n2, n3 = padded_extent(solver.decomp)
    pointwise!(_scale_grad_point!, g, n1, n2, n3, g, ih)
    return g
end

# The flux-divergence accumulation bodies of compute_rhs!: zero one conserved
# component's interior, subtract a divergence, subtract a Jacobian-scaled
# divergence, and form the A_d·F_d product over the full padded array.
@inline function _zero_component_point!(dQ, c, o1, o2, o3, i, j, k)
    @inbounds dQ[i+o1, j+o2, k+o3, c] = 0
    return nothing
end

@inline function _subtract_div_point!(dQ, tmp, c, o1, o2, o3, i, j, k)
    @inbounds dQ[i+o1, j+o2, k+o3, c] -= tmp[i+o1, j+o2, k+o3]
    return nothing
end

@inline function _subtract_jac_div_point!(dQ, tmp, inv_J, c, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        dQ[I, c] -= inv_J[I] * tmp[I]
    end
    return nothing
end

@inline function _area_flux_point!(tmp_b, Ad, F, i, j, k)
    @inbounds tmp_b[i, j, k] = Ad[i, j, k] * F[i, j, k]
    return nothing
end

# The same product with the pressure taken out of the flux, for the radial
# momentum whose pressure term is ∂p/∂r.
@inline function _area_flux_less_point!(tmp_b, Ad, F, p, i, j, k)
    @inbounds tmp_b[i, j, k] = Ad[i, j, k] * (F[i, j, k] - p[i, j, k])
    return nothing
end

# Antipodal signs of velocity and conserved components for the fold (if any)
# on dimension d; scalars, partial densities, and energy are always +1.
function vel_parity(solver::SolverLike, d::Int, j::Int)
    fold = _fold_at(solver, d)
    return fold === nothing ? 1 : fold.sigvel[j]
end
function cons_parity(solver::SolverLike, d::Int, c::Int)
    fold = _fold_at(solver, d)
    return fold === nothing ? 1 : conserved_parity(solver.equations, fold.sigvel, c)
end

assemble_fluxes!(solver::SolverLike, Q) =
    (_assemble_fluxes!(patch_fields(solver), solver.eos, solver.transport,
                       equation_layout(solver.equations),
                       _shared_species_diffusivity(solver), solver.art.species_flux,
                       _sharpening(solver), Q); solver)

# No `::Type` argument here: a `Type` inside `pointwise!`'s Vararg defeats
# Julia's specialization heuristics and the body call turns into a per-point
# runtime dispatch, measured as assemble_fluxes! at 9× its cost. The element
# type comes off an array argument instead.
@inline function _fluxes_point!(Q, eos, rho, u, v, w, p, T_ion,
                                cp_mix, mu_art, beta_art, kappa_art, D_art, Y,
                                grad_u, gT, gY, flux, transport, i_energy, act,
                                bulk, o1, o2, o3, i, j, k)
    T = eltype(rho)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = rho[I]
        uv = (u[I], v[I], w[I])
        pI = p[I]
        Tp = T_ion[I]
        point = _species_point(eos, Tp)
        E = Q[I, i_energy]
        molecular = transport_at(transport, eos, T_ion, rho, cp_mix, Y, I)
        μ = molecular.mu + mu_art[I]
        β = beta_art[I]
        κ = molecular.kappa + kappa_art[I]
        divu = grad_u[1][1][I] + grad_u[2][2][I] + grad_u[3][3][I]
        # T(2)/T(3), not the literal 2/3: the Float64 literal promotes the
        # normal stresses under a narrower T, making τ a heterogeneous tuple
        # whose runtime indexing is a dynamic field access, an InvalidIRError
        # on device. The Float64 value is identical.
        two_thirds = T(2) / T(3)
        τ11 = μ * (2*grad_u[1][1][I] - two_thirds * divu) + β * divu
        τ22 = μ * (2*grad_u[2][2][I] - two_thirds * divu) + β * divu
        τ33 = μ * (2*grad_u[3][3][I] - two_thirds * divu) + β * divu
        τ12 = μ * (grad_u[1][2][I] + grad_u[2][1][I])
        τ13 = μ * (grad_u[1][3][I] + grad_u[3][1][I])
        τ23 = μ * (grad_u[2][3][I] + grad_u[3][2][I])
        τ = ((τ11, τ12, τ13), (τ12, τ22, τ23), (τ13, τ23, τ33))
        # A collapsed dimension's flux is neither exchanged nor differenced,
        # so assembling it was a third of this phase wasted on a planar run.
        act[1] && _fluxes_along!(Val(1), flux[1], gY[1], gT[1], Y, D_art, eos, point,
                                 molecular, bulk, ρ, uv, pI, E, κ, τ[1], I)
        act[2] && _fluxes_along!(Val(2), flux[2], gY[2], gT[2], Y, D_art, eos, point,
                                 molecular, bulk, ρ, uv, pI, E, κ, τ[2], I)
        act[3] && _fluxes_along!(Val(3), flux[3], gY[3], gT[3], Y, D_art, eos, point,
                                 molecular, bulk, ρ, uv, pI, E, κ, τ[3], I)
    end
    return nothing
end

# Left-to-right sum of `acc` and the elements of a tuple: the order of the
# loop `for x in t; acc += x; end`, unrolled.
@inline _sum_in_order(acc, t::Tuple) = _sum_in_order(acc + first(t), Base.tail(t))
@inline _sum_in_order(acc, ::Tuple{}) = acc

# The fluxes along dimension `d` at one point. `fd` holds the species
# fluxes and then the momentum and energy fluxes along `d`, `gYd` the
# mass-fraction gradients along `d`; `Y`, `D_art` and both of these are
# tuples whose length is the species count, so the species sums unroll at
# constant indices.
@inline function _fluxes_along!(::Val{d}, fd, gYd, gTd, Y, D_art, eos, point,
                                molecular, bulk, ρ, uv, pI, E, κ, τd, I) where {d}
    T = typeof(ρ)
    N = length(Y)
    @inbounds begin
        ud = uv[d]
        # Per-species diffusion with a correction velocity:
        # J_k = −ρ D_k ∇Y_k + ρ Y_k V_c, V_c = Σ_j D_j ∇Y_j,
        # which enforces Σ_k J_k = 0 exactly since ΣY_k = 1. Under the
        # shared-D_b species channels (`:partial_density`, `:bulk`) the
        # artificial part of this flux is added afterwards by
        # `_partial_density_flux_point!` or `_bulk_flux_point!`, so only
        # the molecular diffusivity D0 enters here; `D_art` then holds
        # D_b, which must not be read as a Fickian coefficient.
        # Each closure below is a method of its own: the enclosing `@inbounds`
        # does not reach it, and without `@inline` it stays a call per
        # species, a GC safepoint at which every array of the body is rooted
        # again (measured at 25% of this phase).
        Dk = ntuple(Val(N)) do sp
            @inline
            @inbounds bulk ? molecular.D[sp] : molecular.D[sp] + D_art[sp][I]
        end
        Vc = _sum_in_order(zero(T), ntuple(Val(N)) do sp
            @inline
            @inbounds Dk[sp] * gYd[sp][I]
        end)
        hJ = ntuple(Val(N)) do sp
            @inline
            @inbounds begin
                Jkd = ρ * (-Dk[sp] * gYd[sp][I] + Y[sp][I] * Vc)
                fd[sp][I] = ρ * Y[sp][I] * ud + Jkd
                species_enthalpy(eos, sp, point) * Jkd
            end
        end
        hdiff = _sum_in_order(zero(T), hJ)          # Σ_k h_k J_{k,d}
        fd[N + 1][I] = ρ * ud * uv[1] + (d == 1 ? pI : zero(T)) - τd[1]
        fd[N + 2][I] = ρ * ud * uv[2] + (d == 2 ? pI : zero(T)) - τd[2]
        fd[N + 3][I] = ρ * ud * uv[3] + (d == 3 ? pI : zero(T)) - τd[3]
        fd[N + 4][I] = (E + pI) * ud -
                       (uv[1]*τd[1] + uv[2]*τd[2] + uv[3]*τd[3]) -
                       κ * gTd[I] + hdiff
    end
    return nothing
end

# The field collections of one patch as tuples of their arrays, with the
# species count `N` in the type: `flux` and `gY` per dimension, `grad_u` as
# its rows. A body indexing these at the constant indices an unrolled
# species sum produces reads each array through one load of a tuple field.
# The `Vector`/`Matrix` forms instead cost a load of the element, a null
# check and a reload of the array's size per access, and a tuple indexed at
# a runtime species index cost 3× (see `FieldVector`).
@inline _species_tuple(v, ::Val{N}) where {N} = ntuple(sp -> v[sp], Val(N))
@inline _dims_tuple(m, ::Val{N}) where {N} =
    ntuple(d -> ntuple(c -> m[d, c], Val(N)), Val(3))

# Keyed on the field storage, EOS, transport and `Q` only (see `PatchFields`),
# so every scheme, detector, dimensionality and patch wrapper shares one
# compiled body per (T, array type, EOS, transport, species count).
function _assemble_fluxes!(f, eos, tr, eqi, shared::Bool, species_flux::Symbol,
                           sharpen::Bool, Q)
    # A dynamic dispatch on the species count, once per whole-array assembly.
    # The species count is not in the solver type, so solvers differing only
    # in it share every other compiled method; only the launches below compile
    # once per species count.
    _launch_fluxes!(Val(eqi[1]), f, eos, tr, eqi, shared, species_flux, sharpen, Q)
    return nothing
end

function _launch_fluxes!(::Val{N}, f, eos, tr, eqi, shared::Bool, species_flux::Symbol,
                         sharpen::Bool, Q) where {N}
    n_species, n_cons, i_energy, (m1, m2, m3) = eqi
    decomp = f.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    ft = f.ft
    flux = ft.flux
    columns = (ntuple(identity, Val(N))..., m1, m2, m3, i_energy)
    fd = ntuple(d -> map(c -> flux[d, c], columns), Val(3))
    pointwise!(_fluxes_point!, f.rho, nx, ny, nz,
               Q, eos, f.rho, f.u, f.v, f.w, f.p,
               f.T_ion, f.cp_mix, f.mu_art, f.beta_art, f.kappa_art,
               _species_tuple(ft.D_art, Val(N)), _species_tuple(ft.Y, Val(N)),
               _dims_tuple(ft.grad_u, Val(3)), f.grad_T_ion,
               _dims_tuple(ft.grad_Y, Val(N)), fd,
               tr, i_energy, decomp.active, shared, o1, o2, o3)
    # The shared-D_b species channels, D_b read from `D_art[1]` (every
    # `D_art[k]` holds it) and ∂_d Q_c from the gradients `compute_rhs!` filled
    # into `grad_Q`: `:bulk` adds −D_b ∂_d Q_c to every component,
    # `:partial_density` the partial-density flux below. A separate pass rather
    # than a branch inside the body above, which is at the argument count
    # the launcher accepts. `species_flux` is a setup constant identical on
    # every rank, so the branch is safe with no collective below it.
    if shared && species_flux === :bulk
        pointwise!(_bulk_flux_point!, f.rho, nx, ny, nz,
                   flux, f.D_art[1], FieldMatrix(f.grad_Q),
                   n_cons, decomp.active, o1, o2, o3)
    elseif shared && species_flux === :partial_density
        # The sharpening fluxes sit in the `grad_Q` columns past the partial
        # densities, which exist only under `sharpen`; without it the
        # species columns stand in for them, unread.
        gQ = FieldMatrix(f.grad_Q)
        gS = sharpen ? ntuple(d -> ntuple(sp -> gQ[d, N + sp], Val(N)), Val(3)) :
                       _dims_tuple(gQ, Val(N))
        pointwise!(_partial_density_flux_point!, f.rho, nx, ny, nz,
                   fd, eos, f.D_art[1], _dims_tuple(gQ, Val(N)), gS,
                   f.u, f.v, f.w, f.T_ion, decomp.active, sharpen, o1, o2, o3)
    end
    return nothing
end

# The partial-density species channel of Brill, Olson & Bokman (2025, eqs.
# 38–40): J_k = −D_b ∂_d(ρY_k) on each species, the mass flux ΣJ carried into
# momentum as (ΣJ) u and into energy as (ΣJ) |u|²/2 + Σ_k e_k J_k, with the
# species internal energy e_k and not the enthalpy. At uniform (u, p, T) every
# added term is a fixed linear combination of the species fluxes, so the state
# is an exact discrete invariant, and no stress or conduction is added beyond
# what the mass flux carries. Under `sharpen` each J_k first takes the
# sharpening flux S_k that `_sharpening_fluxes!` left in `gQ[d, n_species + k]`,
# so the consistency terms are those of the total species flux.
@inline function _partial_density_flux_point!(flux, eos, D_b, gQ, gS, u, v, w, T_ion,
                                              act, sharpen, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        Db = D_b[I]
        uv = (u[I], v[I], w[I])
        point = _species_point(eos, T_ion[I])
        ke = (uv[1]^2 + uv[2]^2 + uv[3]^2) / 2
        act[1] && _partial_density_along!(flux[1], gQ[1], gS[1], eos, point, Db, uv,
                                          ke, sharpen, I)
        act[2] && _partial_density_along!(flux[2], gQ[2], gS[2], eos, point, Db, uv,
                                          ke, sharpen, I)
        act[3] && _partial_density_along!(flux[3], gQ[3], gS[3], eos, point, Db, uv,
                                          ke, sharpen, I)
    end
    return nothing
end

# The partial-density fluxes along one dimension: `fd` the species, momentum
# and energy fluxes along it, `gQd` the partial-density gradients and `gSd`
# the sharpening fluxes, tuples of the species count as in `_fluxes_along!`.
@inline function _partial_density_along!(fd, gQd, gSd, eos, point, Db, uv, ke, sharpen,
                                         I)
    N = length(gQd)
    @inbounds begin
        J = ntuple(Val(N)) do sp
            @inline
            @inbounds begin
                Jkd = -Db * gQd[sp][I]
                sharpen && (Jkd += gSd[sp][I])
                fd[sp][I] += Jkd
                Jkd
            end
        end
        Jsum = _sum_in_order(zero(Db), J)
        eJ = _sum_in_order(zero(Db), ntuple(Val(N)) do sp
            @inline
            @inbounds _species_internal_energy(eos, sp, point) * J[sp]
        end)
        fd[N + 1][I] += Jsum * uv[1]
        fd[N + 2][I] += Jsum * uv[2]
        fd[N + 3][I] += Jsum * uv[3]
        fd[N + 4][I] += Jsum * ke + eJ
    end
    return nothing
end

# e_k(T) = h_k(T) − R_k T for the ideal-gas models; a single-component
# stiffened gas carries no composition gradient for the channel to act on.
Base.@propagate_inbounds _species_internal_energy(eos, k::Int, T_ion) =
    species_enthalpy(eos, k, T_ion) - eos.Rk[k] * T_ion
@inline _species_internal_energy(eos::Union{StiffenedGas,StiffenedGasCoeffs},
                                 ::Int, T_ion) = eos.cv * T_ion
@inline _species_internal_energy(eos::Nasa9Model, k::Int, point::Nasa9Powers) =
    species_energy(eos, k, point)

@inline function _bulk_flux_point!(flux, D_b, gQ, n_cons, act, o1, o2, o3,
                                   i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        Db = D_b[I]
        for d in 1:3
            act[d] || continue
            for c in 1:n_cons
                flux[d, c][I] -= Db * gQ[d, c][I]
            end
        end
    end
    return nothing
end

# --- The flux divergence through interface ends -------------------------------
#
# Under `interface_flux = :ghost` the inviscid flux of an interface dimension
# is evaluated pointwise over the padded block, ghosts included, from the
# primitives: at an interface end those ghosts hold the neighbour's state as
# of the current stage (the same-level ghost refill, or the coarse-fine shell
# imposed at the stage time), so the gradient plans' interface rows, which
# read ghost layers, apply to it. The molecular flux (viscous stress,
# conduction and species diffusion) needs gradients beyond the block. Its
# interior values go into the patch's `ghost_flux` array here, and its ghost
# values enter afterwards, once every patch of the level has evaluated
# (`_level_ghost_fluxes!`): at a same-level face the neighbour's own interior
# values arrive through the level's flux records, and at a coarse-fine face
# they are evaluated from the shell's gradient ring. The ghost-differenced
# flux of a viscous run is therefore assembled and solved there, once, over
# the whole block; the right-hand side here takes only the remainder. What
# the ghost fluxes do not cover (the artificial fluxes and the wall
# corrections) keeps the one-sided rows of `div_plans`. Without molecular
# transport both solves run here. The split costs one pointwise pass per
# component and interface dimension and, where a remainder exists, a second
# line solve; the molecular part adds one pass over all interface dimensions
# and one halo exchange per dimension. It allocates nothing beyond the
# `ghost_flux` arrays sized at construction; `tmp_b`, `tmp_a` and `sensor_sp`
# are its scratch.
#
# On the axisymmetric cylindrical metric the divergence along `d` is
# inv_J·D(A_d F_d), and both parts carry the product: A_d(F − P) through the
# divergence plans and A_d P through the gradient plans, each scaled by inv_J
# at the interior points. The geometry arrays are analytic over the padded
# extent, so A_d is exact on the ghost layers of an interface end. The radial
# momentum's pressure term is ∂p/∂r through the gradient plans, from the
# pressure's own ghosts, and the flux it leaves carries no pressure.

"Whether dimension `d` of this patch has an interface end the divergence closes."
@inline function _interface_dim(solver::SolverLike, d::Int)
    fold = _fold_at(solver, d)
    fold === nothing && return _distinct_at(solver.div_plans, solver.deriv_plans, d)
    return fold.div_plans !== nothing
end

# `a[d] !== b[d]`, compared at each constant tuple position. Through `_plan_at`
# the two operands are values of a union of the plan type and `Nothing`, which
# `!==` boxes: two heap-allocated plans per call, and a refined patch asks once
# per dimension per conserved component.
@inline _distinct_at(a::Tuple, b::Tuple, d::Int) =
    d == 1 ? a[1] !== b[1] : d == 2 ? a[2] !== b[2] : a[3] !== b[3]

# Whether the flux along `d` carries a part beyond the ghost-differenced one:
# the artificial properties, molecular transport not carried by the ghost
# fluxes, or a face whose `correct_flux!` may rewrite the flux plane. Setup
# constants of the patch, so every rank of its communicator takes the same
# branch.
function _flux_remainder(solver::SolverLike, d::Int)
    molecular = !_zero_molecular_diffusion(solver.transport) && !_ghost_viscous(solver)
    (solver.art.enabled || _shared_species_diffusivity(solver) || molecular) &&
        return true
    bc_lo, bc_hi = solver.bcs[d]
    per = solver.decomp.periodic[d]
    # A symmetry plane or the axis corrects no flux; its fold carries the
    # condition.
    unforced(bc) = per || bc isa Union{InterfaceBC,SymmetryPlaneBC,AxisBC}
    return !unforced(bc_lo) || !unforced(bc_hi)
end

# The area factor and the Jacobian the ghost-differenced divergence along `d`
# takes: `nothing` for both on an unstretched Cartesian grid, where they are
# one and the products are skipped, as in `compute_rhs!`. Dispatched on the
# metric and stretch types, so the result is inferred.
@inline _ghost_geometry(solver::SolverLike, d::Int) =
    _ghost_geometry(solver.metric, solver.stretch, solver, d)
@inline _ghost_geometry(::CartesianMetric, ::NTuple{3,Nothing}, solver, d::Int) =
    (nothing, nothing)
@inline _ghost_geometry(::Metric, stretch, solver, d::Int) =
    (solver.area_d[d], solver.inv_J)

# `f` times the area factor at `I`, or `f` itself on unit geometry.
@inline _area_scaled(::Nothing, I, f) = f
@inline _area_scaled(Ad, I, f) = @inbounds Ad[I] * f

# The ghost-differenced flux of component `c` along `d` at every padded point
# into `out`: the inviscid flux, plus the molecular flux `G` under `viscous`,
# or, under `remainder`, the assembled flux `F` less it, either one times the
# area factor `Ad` where it is not `nothing`. The inviscid expressions are
# those of `_fluxes_point!`, so on an inviscid run the remainder is exactly
# zero. At an interface end `F`'s ghosts hold no data and neither does the
# remainder there; the divergence rows that take it read none, and `G` is
# zero there (`_molecular_flux_point!`). Under `less_p` the ghost-differenced
# flux leaves out the pressure, which the radial momentum of the r-z and
# spherical metrics and the spherical θ-momentum take as a gradient
# (`_pressure_gradient`); the remainder does not.
@inline function _inviscid_flux_point!(out, F, Q, rho, u, v, w, p, Y, G, Ad, c, d,
                                       n_species, m1, m2, m3, i_energy,
                                       remainder, viscous, less_p, i, j, k)
    @inbounds begin
        I = CartesianIndex(i, j, k)
        f = _ghost_differenced_flux(Q, rho, u, v, w, p, Y, G, c, d, n_species,
                                    m1, m2, m3, i_energy, viscous, I)
        out[I] = _area_scaled(Ad, I, remainder ? F[I] - f : ifelse(less_p, f - p[I], f))
    end
    return nothing
end

# The inviscid form of `_inviscid_flux_point!` with the component's kind fixed
# at the launch, one body per kind: a partial density, a momentum component
# (`along` when its direction is `d`, where the pressure enters) and the
# energy. `ud` is the velocity component along `d`. Each writes the
# ghost-differenced flux into `out` and, unless `rem` is `nothing`, the
# remainder into `rem`, by the expressions of `_ghost_differenced_flux`, so
# the values are bitwise those of the generic body, which selects the
# expression at every point. The body adds the offsets `o1`, `o2`, `o3` to its
# index, so that a launch may cover part of the padded block (`_ghost_split!`).
@inline function _species_split_point!(rem, out, F, rho, Yc, ud, p, Ad, less_p,
                                       o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        _store_split!(rem, out, F, p, Ad, rho[I] * Yc[I] * ud[I], less_p, I)
    end
    return nothing
end

@inline function _momentum_split_point!(rem, out, F, rho, ud, um, p, Ad, along, less_p,
                                        o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        pI = p[I]
        f = rho[I] * ud[I] * um[I] + ifelse(along, pI, zero(pI))
        _store_split!(rem, out, F, p, Ad, f, less_p, I)
    end
    return nothing
end

@inline function _energy_split_point!(rem, out, F, Q, ud, p, Ad, i_energy, less_p,
                                      o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        _store_split!(rem, out, F, p, Ad, (Q[I, i_energy] + p[I]) * ud[I], less_p, I)
    end
    return nothing
end

@inline function _store_split!(rem, out, F, p, Ad, f, less_p, I)
    @inbounds begin
        rem === nothing || (rem[I] = _area_scaled(Ad, I, F[I] - f))
        out[I] = _area_scaled(Ad, I, ifelse(less_p, f - p[I], f))
    end
    return nothing
end

"""
    _ghost_split!(rem, out, solver, F, Q, c, d, Ad, less_p)

The ghost-differenced inviscid flux of component `c` along `d` into `out`
and, unless `rem` is `nothing`, the remainder `F − f` into `rem`, each times
the area factor `Ad` where it is not `nothing`, by the kind's body above. The
launch covers the padded extent along `d` and the interior across it, since
the line fills of the divergence and gradient plans, host and device, read a
field only on the lines through interior transverse nodes; on a 3-D tile of
25 nodes a side with four ghost layers that leaves out 43% of the padded
block. A fold along `d` takes the whole padded block, as its mirror fill and
butterfly read the field there. A component outside the Navier–Stokes layout
takes the generic body (`_generic_split!`).
"""
function _ghost_split!(rem, out, solver::SolverLike, F, Q, c::Int, d::Int, Ad,
                       less_p::Bool)
    decomp = solver.decomp
    eq = solver.equations
    nl = decomp.n_local
    pad = decomp.n_halo_d
    nf = padded_extent(decomp)
    whole = _fold_at(solver, d) !== nothing
    b1, b2, b3 = ntuple(e -> whole || e == d ? nf[e] : nl[e], 3)
    o1, o2, o3 = ntuple(e -> whole || e == d ? 0 : pad[e], 3)
    vel = (solver.u, solver.v, solver.w)
    m1, m2, m3 = eq.i_mom
    if c <= eq.n_species
        pointwise!(_species_split_point!, out, b1, b2, b3, rem, out, F, solver.rho,
                   solver.Y[c], vel[d], solver.p, Ad, less_p, o1, o2, o3)
    elseif c == m1 || c == m2 || c == m3
        m = c == m1 ? 1 : c == m2 ? 2 : 3
        pointwise!(_momentum_split_point!, out, b1, b2, b3, rem, out, F, solver.rho,
                   vel[d], vel[m], solver.p, Ad, d == m, less_p, o1, o2, o3)
    elseif c == eq.i_energy
        pointwise!(_energy_split_point!, out, b1, b2, b3, rem, out, F, Q, vel[d],
                   solver.p, Ad, eq.i_energy, less_p, o1, o2, o3)
    else
        _generic_split!(rem, out, solver, F, Q, c, d, Ad, less_p)
    end
    return nothing
end

# `_ghost_split!` through `_inviscid_flux_point!`, one pass per field over the
# whole padded block.
function _generic_split!(rem, out, solver::SolverLike, F, Q, c::Int, d::Int, Ad,
                         less_p::Bool)
    eq = solver.equations
    m1, m2, m3 = eq.i_mom
    n1f, n2f, n3f = padded_extent(solver.decomp)
    for (dest, remainder) in ((rem, true), (out, false))
        dest === nothing && continue
        pointwise!(_inviscid_flux_point!, dest, n1f, n2f, n3f,
                   dest, F, Q, solver.rho, solver.u, solver.v, solver.w,
                   solver.p, solver.field_tuples.Y, solver.ghost_flux[d], Ad, c, d,
                   eq.n_species, m1, m2, m3, eq.i_energy, remainder, false, less_p)
    end
    return nothing
end

@inline function _ghost_differenced_flux(Q, rho, u, v, w, p, Y, G, c, d, n_species,
                                         m1, m2, m3, i_energy, viscous, I)
    T = eltype(rho)
    @inbounds begin
        ρ = rho[I]
        uv = (u[I], v[I], w[I])
        ud = uv[d]
        pI = p[I]
        f = zero(T)
        if c <= n_species
            f = ρ * Y[c][I] * ud
        elseif c == m1
            f = ρ * ud * uv[1] + (d == 1 ? pI : zero(T))
        elseif c == m2
            f = ρ * ud * uv[2] + (d == 2 ? pI : zero(T))
        elseif c == m3
            f = ρ * ud * uv[3] + (d == 3 ? pI : zero(T))
        elseif c == i_energy
            f = (Q[I, i_energy] + pI) * ud
        end
        viscous && (f += G[I, c])
        return f
    end
end

# The molecular flux along `d` of every conserved component into `G[I, :]`:
# the viscous stress, the conduction and the species diffusion of
# `_fluxes_point!` with the artificial coefficients left out, from the
# velocity gradients `gu` (`gu[j][m]` is ∂_j u_m), the temperature gradient
# `dTd` along `d` and `dYd(sp)`, the mass-fraction gradient along `d`. The
# expressions are `_fluxes_point!`'s term for term, so with the artificial
# properties off the two agree to the association of the energy sum.
@inline function _molecular_flux!(G, I, eos, ρ, uv, Tp, Y, molecular, gu, dTd,
                                  dYd::F, n_species, m1, m2, m3, i_energy,
                                  d) where {F}
    T = typeof(ρ)
    @inbounds begin
        μ = molecular.mu
        κ = molecular.kappa
        divu = gu[1][1] + gu[2][2] + gu[3][3]
        two_thirds = T(2) / T(3)
        τ11 = μ * (2*gu[1][1] - two_thirds * divu)
        τ22 = μ * (2*gu[2][2] - two_thirds * divu)
        τ33 = μ * (2*gu[3][3] - two_thirds * divu)
        τ12 = μ * (gu[1][2] + gu[2][1])
        τ13 = μ * (gu[1][3] + gu[3][1])
        τ23 = μ * (gu[2][3] + gu[3][2])
        τd = d == 1 ? (τ11, τ12, τ13) : d == 2 ? (τ12, τ22, τ23) : (τ13, τ23, τ33)
        Vc = zero(T)
        for sp in 1:n_species
            Vc += molecular.D[sp] * dYd(sp)
        end
        hdiff = zero(T)
        point = _species_point(eos, Tp)
        for sp in 1:n_species
            Jkd = ρ * (-molecular.D[sp] * dYd(sp) + Y[sp][I] * Vc)
            G[I, sp] = Jkd
            hdiff += species_enthalpy(eos, sp, point) * Jkd
        end
        G[I, m1] = -τd[1]
        G[I, m2] = -τd[2]
        G[I, m3] = -τd[3]
        G[I, i_energy] = -(uv[1]*τd[1] + uv[2]*τd[2] + uv[3]*τd[3]) -
                         κ * dTd + hdiff
    end
    return nothing
end

# The molecular flux along each dimension flagged in `dims` into `G1`, `G2`,
# `G3` over the padded block from the interior gradients: evaluated where the
# point is interior along every active dimension, zero elsewhere, so an
# interface end's ghost layers start from zero and the rank halos along `d`
# take the neighbour's values in the exchange that follows. One pass serves
# every interface dimension, which reads the transport coefficients and the
# gradients once. `stride` is the padded extent along the third dimension,
# which a stacked launch adds to `k` once per tile.
@inline function _molecular_flux_point!(G1, G2, G3, dims, eos, rho, u, v, w, T_ion,
                                        cp_mix, Y, grad_u, gT, gY, transport,
                                        n_species, m1, m2, m3, i_energy, n_cons,
                                        nl, pad, stride, i, j, k)
    T = eltype(rho)
    @inbounds begin
        I = CartesianIndex(i, j, k)
        kl = (k - 1) % stride + 1
        inside = pad[1] < i <= pad[1] + nl[1] && pad[2] < j <= pad[2] + nl[2] &&
                 pad[3] < kl <= pad[3] + nl[3]
        if !inside
            for c in 1:n_cons
                dims[1] && (G1[I, c] = zero(T))
                dims[2] && (G2[I, c] = zero(T))
                dims[3] && (G3[I, c] = zero(T))
            end
            return nothing
        end
        molecular = transport_at(transport, eos, T_ion, rho, cp_mix, Y, I)
        gu = ((grad_u[1, 1][I], grad_u[1, 2][I], grad_u[1, 3][I]),
              (grad_u[2, 1][I], grad_u[2, 2][I], grad_u[2, 3][I]),
              (grad_u[3, 1][I], grad_u[3, 2][I], grad_u[3, 3][I]))
        ρ = rho[I]
        uv = (u[I], v[I], w[I])
        Tp = T_ion[I]
        dims[1] && _molecular_flux!(G1, I, eos, ρ, uv, Tp, Y, molecular, gu, gT[1][I],
                                    sp -> gY[1, sp][I], n_species, m1, m2, m3,
                                    i_energy, 1)
        dims[2] && _molecular_flux!(G2, I, eos, ρ, uv, Tp, Y, molecular, gu, gT[2][I],
                                    sp -> gY[2, sp][I], n_species, m1, m2, m3,
                                    i_energy, 2)
        dims[3] && _molecular_flux!(G3, I, eos, ρ, uv, Tp, Y, molecular, gu, gT[3][I],
                                    sp -> gY[3, sp][I], n_species, m1, m2, m3,
                                    i_energy, 3)
    end
    return nothing
end

# The dimensions `_ghost_flux_divergence!` serves: active, with an interface
# end the divergence closes. Setup constants of the patch.
_ghost_flux_dims(solver::SolverLike) =
    ntuple(d -> solver.decomp.active[d] && _interface_dim(solver, d), 3)

# Phase one of the molecular ghost flux: the interior molecular flux along
# every interface dimension `d` into the patch's `ghost_flux[d]`, with its
# rank halos along `d` exchanged, from the gradients this evaluation
# computed. Collective over the patch's communicator.
function _molecular_ghost_flux!(solver::SolverLike, Q)
    decomp = solver.decomp
    eq = solver.equations
    m1, m2, m3 = eq.i_mom
    n1f, n2f, n3f = padded_extent(decomp)
    ft = solver.field_tuples
    dims = _ghost_flux_dims(solver)
    G1, G2, G3 = solver.ghost_flux
    route = dims[1] ? G1 : dims[2] ? G2 : G3
    _ghost_viscous(solver) &&
        pointwise!(_molecular_flux_point!, route, n1f, n2f, n3f,
                   G1, G2, G3, dims, solver.eos, solver.rho, solver.u, solver.v,
                   solver.w, solver.T_ion, solver.cp_mix, ft.Y, ft.grad_u,
                   solver.grad_T_ion, ft.grad_Y, solver.transport, eq.n_species,
                   m1, m2, m3, eq.i_energy, eq.n_cons, decomp.n_local,
                   decomp.n_halo_d, n3f)
    for d in 1:3
        _remainder_ghosted(solver, d) || continue
        G = solver.ghost_flux[d]
        for c in 1:eq.n_cons
            pointwise!(_remainder_flux_point!, G, n1f, n2f, n3f,
                       G, solver.flux[d, c], Q, solver.rho, solver.u, solver.v,
                       solver.w, solver.p, ft.Y, c, d, eq.n_species, m1, m2, m3,
                       eq.i_energy, decomp.n_local, decomp.n_halo_d, n3f)
        end
    end
    for d in 1:3
        dims[d] && size(solver.ghost_flux[d], 4) > 0 &&
            exchange_dim_batch!(ComponentViews(solver.ghost_flux[d]), decomp, d)
    end
    return solver
end

# Whether dimension `d` takes its whole flux through the gradient plans, the
# remainder F − f carried in `ghost_flux` (`GHOST_FLUX_REMAINDER`): an
# interface dimension whose interface ends are all same-level.
function _remainder_ghosted(solver::SolverLike, d::Int)
    _ghost_remainder(solver) || return false
    solver.decomp.active[d] && _interface_dim(solver, d) || return false
    size(solver.ghost_flux[d], 4) > 0 || return false
    GHOST_REMAINDER_EXTRAPOLATE[] && return true
    bc_lo, bc_hi = solver.bcs[d]
    return !parent_fed(bc_lo) && !parent_fed(bc_hi)
end

# The remainder's ghost layers at each coarse-fine face of `d` this rank holds,
# by polynomial extrapolation of the interior values along the line. Phase
# two, per tile: a stacked level's spanning patch carries the first tile's
# face kinds only.
function _extrapolate_coarse_fine!(solver::SolverLike, d::Int)
    decomp = solver.decomp
    pad = decomp.n_halo_d
    nl = decomp.n_local
    G = solver.ghost_flux[d]
    deg = min(GHOST_REMAINDER_DEGREE[], nl[d] - 1)
    o1, o2 = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
    for side in 1:2
        parent_fed(solver.bcs[d][side]) && _ghost_face(solver, d, side) || continue
        base = side == 1 ? 0 : pad[d] + nl[d]
        edge = side == 1 ? pad[d] + 1 : pad[d] + nl[d]
        dir = side == 1 ? 1 : -1
        pointwise!(_extrapolate_ghost_point!, G, pad[d], nl[o1], nl[o2],
                   G, size(G, 4), d, base, edge, dir, pad, deg)
    end
    return G
end

# Lagrange extrapolation of every component to a ghost node `m` spacings
# beyond the edge node, from the edge node and the `degree` nodes inward.
@inline function _extrapolate_ghost_point!(G, n_cons, d, base, edge, dir, pad,
                                           degree, a, b, c3)
    T = eltype(G)
    @inbounds begin
        o1, o2 = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
        idx = ntuple(e -> e == d ? base + a : e == o1 ? pad[e] + b : pad[e] + c3, 3)
        m = abs(idx[d] - edge)
        for c in 1:n_cons
            s = zero(T)
            for j in 0:degree
                w = one(T)
                for k in 0:degree
                    k == j && continue
                    w *= T(-m - k) / T(j - k)
                end
                src = ntuple(e -> e == d ? edge + dir * j : idx[e], 3)
                s += w * G[CartesianIndex(src), c]
            end
            G[CartesianIndex(idx), c] = s
        end
    end
    return nothing
end

# The remainder F − f of component `c` along `d` into `G[I, c]` at points
# interior along every active dimension, zero elsewhere, as
# `_molecular_flux_point!` lays out the molecular flux.
@inline function _remainder_flux_point!(G, F, Q, rho, u, v, w, p, Y, c, d,
                                        n_species, m1, m2, m3, i_energy,
                                        nl, pad, stride, i, j, k)
    T = eltype(rho)
    @inbounds begin
        I = CartesianIndex(i, j, k)
        kl = (k - 1) % stride + 1
        inside = pad[1] < i <= pad[1] + nl[1] && pad[2] < j <= pad[2] + nl[2] &&
                 pad[3] < kl <= pad[3] + nl[3]
        if !inside
            G[I, c] = zero(T)
            return nothing
        end
        f = _ghost_differenced_flux(Q, rho, u, v, w, p, Y, G, c, d, n_species,
                                    m1, m2, m3, i_energy, false, I)
        G[I, c] = F[I] - f
    end
    return nothing
end

# dQ[:, c] -= D_div(F - P) + D_ext(P) along `d`, P the ghost-differenced
# flux: the first through the divergence plans, skipped where the remainder
# is identically zero, the second through the gradient plans, reading the
# ghost fluxes. Both are collective line solves along `d`, and the branches
# are setup constants of the patch. `F` itself is left as assembled. The
# first component's call along the first interface dimension fills the
# molecular flux of every component along every interface dimension
# (`_molecular_ghost_flux!`), which the conserved-component loop of
# `compute_rhs!` reaches before any other. With molecular transport the
# second solve waits for phase two, whose ghost layers of `G` it reads.
function _ghost_flux_divergence!(dQ, c::Int, Fdc, solver::SolverLike, Q, d::Int)
    decomp = solver.decomp
    eq = solver.equations
    m1, m2, m3 = eq.i_mom
    n1f, n2f, n3f = padded_extent(decomp)
    viscous = _ghost_viscous(solver)
    (viscous || _ghost_remainder(solver)) && c == 1 &&
        d == findfirst(_ghost_flux_dims(solver)) && _molecular_ghost_flux!(solver, Q)
    G = solver.ghost_flux[d]
    Y = solver.field_tuples.Y
    Ad, iJ = _ghost_geometry(solver, d)
    # The flux's sign across a fold at the dimension's other end (a refined
    # patch on a symmetry plane or the r-z axis); 1 without one.
    σ = _flux_sign(solver, d, c)
    # The pressure term as a gradient, through the gradient plans, whose
    # interface rows read the pressure's ghosts; p is even across the axis,
    # the origin and the poles. Ahead of the early return below, since it is
    # a line solve every rank of the patch takes.
    less_p = _pressure_gradient(solver, d, c)
    less_p && _ext_subtract_along!(dQ, c, solver.p, solver, d, solver.inv_h[d], 1)
    # The whole flux waits for phase two (`GHOST_FLUX_REMAINDER`).
    _remainder_ghosted(solver, d) && return dQ
    if viscous
        _flux_remainder(solver, d) || return dQ
        pointwise!(_inviscid_flux_point!, solver.tmp_b, n1f, n2f, n3f,
                   solver.tmp_b, Fdc, Q, solver.rho, solver.u, solver.v, solver.w,
                   solver.p, Y, G, Ad, c, d, eq.n_species, m1, m2, m3,
                   eq.i_energy, true, true, less_p)
        div_subtract_along!(dQ, c, solver.tmp_b, solver, d, σ, iJ)
    elseif !_flux_remainder(solver, d)
        _ghost_split!(nothing, solver.tmp_b, solver, Fdc, Q, c, d, Ad, less_p)
        _ext_subtract_along!(dQ, c, solver.tmp_b, solver, d, iJ, σ)
    else
        # Both fields in one pass. `sensor_sp`, dead once the coefficients
        # are assembled (`compute_artificial!`), holds the second until its
        # solve, since a device plan or a fold solves through `tmp_a`.
        _ghost_split!(solver.tmp_b, solver.sensor_sp, solver, Fdc, Q, c, d, Ad, less_p)
        div_subtract_along!(dQ, c, solver.tmp_b, solver, d, σ, iJ)
        _ext_subtract_along!(dQ, c, solver.sensor_sp, solver, d, iJ, σ)
    end
    return dQ
end

# The sign of component `c`'s flux product across the fold on `d`; 1 without
# one.
function _flux_sign(solver::SolverLike, d::Int, c::Int)
    fold = _fold_at(solver, d)
    return fold === nothing ? 1 : fold.sigflux[c]
end

# dQ[:, c] -= inv_J·D_ext(f) along `d` through the gradient plans, whose
# interface rows read `f`'s ghost layers, with `inv_J === nothing` on unit
# geometry; a device plan or a fold takes the two-pass route through `tmp_a`,
# the fold with `f`'s sign `σf` across it.
@noinline function _ext_subtract_along!(dQ, c::Int, f, solver::SolverLike, d::Int, inv_J,
                              σf::Int)
    decomp = solver.decomp
    fold = _fold_at(solver, d)
    _reflux_open!(solver, dQ, c, d, inv_J)
    if fold !== nothing || _device_plan_at(solver.deriv_plans, d)
        fold === nothing ?
            apply_along!(solver.tmp_a, _operator_plan(solver.deriv_plans, d), f, decomp) :
            fold_apply!(solver.tmp_a, f, solver, fold, σf, Val(:deriv))
        nx, ny, nz = decomp.n_local
        o1, o2, o3 = decomp.n_halo_d
        if inv_J === nothing
            pointwise!(_subtract_div_point!, dQ, nx, ny, nz,
                       dQ, solver.tmp_a, c, o1, o2, o3)
        else
            pointwise!(_subtract_jac_div_point!, dQ, nx, ny, nz,
                       dQ, solver.tmp_a, inv_J, c, o1, o2, o3)
        end
    else
        apply_along_subtract!(dQ, c, _operator_plan(solver.deriv_plans, d), f, decomp,
                              inv_J)
    end
    _mask_child_divergence!(dQ, c, f, solver, d, σf, inv_J, Val(:deriv))
    _reflux_close!(solver, dQ, c, f, d, inv_J)
    return dQ
end

# --- Phase two: the molecular flux through interface ends ---------------------
#
# Run by `_level_rhs!` after every patch of a level has evaluated its
# right-hand side, so each patch's `ghost_flux` holds its interior molecular
# flux. The level's flux records copy each same-level neighbour's interior
# values into the abutting ghost layers, over the same records that refill the
# state's ghosts; each coarse-fine face's ghost layers are evaluated from the
# shell's gradient ring. The ghost-differenced flux, the inviscid part from the
# patch's primitives and the molecular part from `ghost_flux`, then goes
# through the gradient plans over the whole block and is subtracted from `dQ`,
# one line solve per component and interface dimension. The primitives are the
# patch's own and the state is unchanged since its right-hand side, so they
# still describe the evaluated state; the assembled flux, which lives on the
# shared workspace, is not read.

# Whether the patch's face `side` of `d` is an interface end whose ghost
# layers this rank holds: a same-level or coarse-fine face (both are
# `InterfaceBC`s), at the patch's own edge of the decomposition.
@inline function _ghost_face(solver::SolverLike, d::Int, side::Int)
    bc = solver.bcs[d][side]
    bc isa InterfaceBC || return false
    decomp = solver.decomp
    return side == 1 ? at_lo_edge(decomp, d) : at_hi_edge(decomp, d)
end

# The molecular flux along `d` on one coarse-fine face's ghost layers, from
# the patch's state and primitives there (the imposed shell) and the gradient
# ring: the primitive gradients follow from the conserved ones by the chain
# rule, ∂u = (∂(ρu) − u ∂ρ)/ρ, ∂Y_k = (∂(ρY_k) − Y_k ∂ρ)/ρ, and ∂e from ∂E,
# with ∂T from ∂e and the ∂Y_k through `_temperature_gradient`. On the
# axisymmetric cylindrical metric `inv_r` adds the curvature terms of the
# velocity gradient that `metric_correct_gradients!` adds in the interior
# (the ring holds the r and z derivatives, which are physical there, and
# nothing along the collapsed θ); it is `nothing` on a Cartesian grid. The
# launch box is (ghost layers along `d`) × (interior transverse), `base` the
# padded index before the first layer.
@inline function _coarse_fine_flux_point!(G, Q, eos, rho, u, v, w, p, T_ion, cp_mix,
                                          Y, gring, table, off, pad, transport, inv_r,
                                          n_species, m1, m2, m3, i_energy, d, base,
                                          a, b, c3)
    T = eltype(rho)
    @inbounds begin
        # The launch coordinates (a along `d`, then the transverse pair in
        # ascending dimension order) as a padded index.
        o1, o2 = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
        idx = ntuple(e -> e == d ? base + a : e == o1 ? pad[e] + b : pad[e] + c3, 3)
        I = CartesianIndex(idx)
        at = _ring_offset(table, idx[1] - pad[1] + off[1], idx[2] - pad[2] + off[2],
                          idx[3] - pad[3] + off[3])
        ρ = rho[I]
        uv = (u[I], v[I], w[I])
        E = Q[I, i_energy]
        drho = ntuple(3) do j
            s = zero(T)
            for sp in 1:n_species
                s += gring[at, 3 * (sp - 1) + j]
            end
            s
        end
        mom = (m1, m2, m3)
        gu = _curvature_corrected(
            ntuple(j -> ntuple(m -> (gring[at, 3 * (mom[m] - 1) + j] -
                                     uv[m] * drho[j]) / ρ, 3), 3), inv_r, uv, I)
        dYd(sp) = (gring[at, 3 * (sp - 1) + d] - Y[sp][I] * drho[d]) / ρ
        de = (gring[at, 3 * (i_energy - 1) + d] - (E / ρ) * drho[d]) / ρ -
             (uv[1] * gu[d][1] + uv[2] * gu[d][2] + uv[3] * gu[d][3])
        Tp = T_ion[I]
        eY = zero(T)
        point = _species_point(eos, Tp)
        for sp in 1:n_species
            eY += _species_internal_energy(eos, sp, point) * dYd(sp)
        end
        dTd = _temperature_gradient(eos, ρ, p[I], Tp, cp_mix[I], drho[d], de, eY)
        molecular = transport_at(transport, eos, T_ion, rho, cp_mix, Y, I)
        _molecular_flux!(G, I, eos, ρ, uv, Tp, Y, molecular, gu, dTd, dYd,
                         n_species, m1, m2, m3, i_energy, d)
    end
    return nothing
end

# The velocity gradient `gu` (`gu[j][m]` is ∂_j u_m) with the cylindrical
# curvature terms of `_grad_corr_cyl_point!`, or unchanged under `nothing`.
@inline _curvature_corrected(gu, ::Nothing, uv, I) = gu
@inline function _curvature_corrected(gu, inv_r, uv, I)
    @inbounds ir = inv_r[I]
    return (gu[1], (gu[2][1] - uv[2] * ir, gu[2][2] + uv[1] * ir, gu[2][3]), gu[3])
end

# The inverse radius the coarse-fine molecular flux takes: the cylindrical
# metric's, `nothing` on a Cartesian grid. A refined level carries no other
# metric (`Solver`).
_ghost_inv_r(solver::SolverLike) = _ghost_inv_r(solver.metric, solver)
_ghost_inv_r(::CartesianMetric, solver) = nothing
_ghost_inv_r(::CylindricalMetric, solver) = solver.inv_r

# ∂T along a line from ∂ρ, ∂e and Σ_k e_k ∂Y_k: for a mixture whose internal
# energy is Σ_k Y_k e_k(T), ∂e = c_v ∂T + Σ_k e_k ∂Y_k; the stiffened gas
# carries e = c_v T + p∞/ρ. `mixture_cv` is the heat capacity the EOS
# contract gives from the primitives.
@inline _temperature_gradient(eos, ρ, p, T_ion, cp, drho, de, eY) =
    (de - eY) / mixture_cv(eos, ρ, p, T_ion, cp)
@inline _temperature_gradient(eos::Union{StiffenedGas,StiffenedGasCoeffs}, ρ, p,
                              T_ion, cp, drho, de, eY) =
    (de + eos.p_inf * drho / (ρ * ρ)) / eos.cv

# The EOS models whose internal energy `_temperature_gradient` inverts; a
# coarse-fine face's molecular ghost flux needs one of them.
_ghost_gradient_eos(eos) =
    eos isa Union{IdealMixture,Nasa9Mixture,StiffenedGas}

# Phase two over one level. Collective over the level's communicator (the
# flux records) and over each patch's (its line solves); every rank that
# evaluated the level's right-hand sides enters it.
function _level_ghost_fluxes!(solver::Solver, lev::Level, states, dQs, comm)
    patches = getfield(solver, :patches)
    isempty(lev.patches) && return dQs
    n_cons = solver.equations.n_cons
    # The same-level records: the root's are the solver's (a slab layout,
    # so they run along one dimension), a refined level's its own per
    # dimension.
    for d in 1:3
        records = lev.index == 0 ? (solver.ghost_sends, solver.ghost_recvs) :
                  (lev.ghost_sends[d], lev.ghost_recvs[d])
        lev.index == 0 && !any(p -> size(p.ghost_flux[d], 4) > 0,
                               view(patches, lev.patches)) && continue
        lev.index > 0 && !lev.phases[d] && continue
        # Wrapped as states are, so that a stacked tile's view, not the
        # stack beneath it, is the array the records index.
        fields = [ConservedState(p.ghost_flux[d]) for p in patches]
        _exchange_ghosts!(solver, fields, comm, records...)
    end
    for (k, pi) in enumerate(lev.patches)
        lt = lev.index == 0 ? nothing : lev.transfers[lev.tiles[k]]
        lt === nothing && continue
        ps = PatchSolver(solver, patches[pi])
        for d in 1:3
            _remainder_ghosted(ps, d) && _extrapolate_coarse_fine!(ps, d)
        end
        lt.gradients === nothing ||
            _coarse_fine_ghost_fluxes!(ps, lt, patches[pi].level_scratch, states[pi])
    end
    # The solves run as the right-hand sides did: per patch, or per stack.
    if isempty(lev.stacks)
        for pi in lev.patches
            _ghost_flux_solves!(PatchSolver(solver, patches[pi]), states[pi],
                                dQs[pi], n_cons)
        end
    else
        for st in lev.stacks
            _ghost_flux_solves!(PatchSolver(solver, st.patch), _stack_state(st, states),
                                _stack_state(st, dQs), n_cons)
        end
    end
    return dQs
end

# The coarse-fine faces' ghost layers of `ghost_flux`, from the gradient ring.
# `scratch` is the patch's `LevelScratch`, whose gradient ring a device patch
# reads in place of the transfer's host one. Every rank of the patch takes the
# same branches: the conditions are the patch's, and `_ghost_face` only
# restricts the writes to the edge.
function _coarse_fine_ghost_fluxes!(solver::SolverLike, lt, scratch, Q)
    decomp = solver.decomp
    eq = solver.equations
    m1, m2, m3 = eq.i_mom
    pad = decomp.n_halo_d
    nl = decomp.n_local
    ft = solver.field_tuples
    dims = _ghost_flux_dims(solver)
    for d in 1:3
        G = solver.ghost_flux[d]
        dims[d] && size(G, 4) > 0 || continue
        # The remainder's extrapolated ghosts carry the molecular part too.
        _remainder_ghosted(solver, d) && continue
        gring = _device_path(G) ? scratch.gring : lt.gradients.gring
        for side in 1:2
            parent_fed(solver.bcs[d][side]) && _ghost_face(solver, d, side) ||
                continue
            o1, o2 = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
            base = side == 1 ? 0 : pad[d] + nl[d]
            pointwise!(_coarse_fine_flux_point!, G, pad[d], nl[o1], nl[o2],
                       G, Q, solver.eos, solver.rho,
                       solver.u, solver.v, solver.w, solver.p, solver.T_ion,
                       solver.cp_mix, ft.Y, gring, (lt.shell.table...,),
                       decomp.offset, pad, solver.transport, _ghost_inv_r(solver),
                       eq.n_species, m1, m2, m3, eq.i_energy, d, base)
        end
    end
    return solver
end

# dQ[:, c] -= D_ext(P) along every interface dimension, P the ghost-differenced
# flux with the molecular part read from `ghost_flux`, ghost layers included.
# Collective over the patch's communicator.
function _ghost_flux_solves!(solver::SolverLike, Q, dQ, n_cons::Int)
    decomp = solver.decomp
    eq = solver.equations
    m1, m2, m3 = eq.i_mom
    n1f, n2f, n3f = padded_extent(decomp)
    ft = solver.field_tuples
    dims = _ghost_flux_dims(solver)
    viscous = _ghost_viscous(solver)
    for d in 1:3
        G = solver.ghost_flux[d]
        dims[d] && size(G, 4) > 0 || continue
        viscous || _remainder_ghosted(solver, d) || continue
        Ad, iJ = _ghost_geometry(solver, d)
        for c in 1:n_cons
            # The pressure gradient was subtracted in phase one.
            pointwise!(_inviscid_flux_point!, solver.tmp_b, n1f, n2f, n3f,
                       solver.tmp_b, solver.flux[d, c], Q, solver.rho, solver.u,
                       solver.v, solver.w, solver.p, ft.Y, G, Ad, c, d, eq.n_species,
                       m1, m2, m3, eq.i_energy, false, true,
                       _pressure_gradient(solver, d, c))
            _ext_subtract_along!(dQ, c, solver.tmp_b, solver, d, iJ,
                                 _flux_sign(solver, d, c))
        end
    end
    return dQ
end

# Entries for the `_cold` calls in `compute_rhs!`. They take the arrays under
# the `ConservedState` wrappers, which are already on the heap, and rewrap
# them here: the immutable wrapper itself would be boxed crossing the dynamic
# call, 16 B per argument per call. A `PatchSolver` crosses as its root solver
# and its patch (`_root_part`, `_patch_part`), both mutable and passed as the
# references they are, and is rejoined here; the wrapper would be boxed at
# each call, 32 B, sixteen times per right-hand side on a tile of a 2-D level.
# Both parts cross behind `_cold`: with the patch's type known at the call,
# inference follows the rejoined, partly typed `PatchSolver` through the whole
# callee when the caller is compiled, which nearly doubled the inference of a
# tile's right-hand side.
_cold_bulk_gradients!(root, patch, q, level_smoothed::Bool) =
    _bulk_gradients!(_rejoin(root, patch), ConservedState(q), level_smoothed)
_cold_ghost_flux_divergence!(dq, c::Int, Fdc, root, patch, q, d::Int) =
    _ghost_flux_divergence!(ConservedState(dq), c, Fdc, _rejoin(root, patch),
                            ConservedState(q), d)

@inline _root_part(solver::Solver) = solver
@inline _root_part(ps::PatchSolver) = getfield(ps, :solver)
@inline _patch_part(solver::Solver) = nothing
@inline _patch_part(ps::PatchSolver) = getfield(ps, :patch)
_rejoin(solver, ::Nothing) = solver
_rejoin(solver, patch) = PatchSolver(solver, patch)

@inline function _copy_component_point!(dest, Q, c, i, j, k)
    @inbounds dest[i, j, k] = Q[i, j, k, c]
    return nothing
end

# The mass-fraction gradients `grad_Y`. Two terms read them: the molecular part
# of the species flux, which multiplies them by the molecular diffusivity, and
# the transverse terms of `NSCBCInflowBC`. Under a shared-D_b species channel
# with `ConstantTransport(mu0 = 0)` the first is identically zero, so `compute_rhs!`
# skips the n_species line solves per direction and the inflow condition takes
# them itself (`correct_rhs!`, above its early return). The flux body then
# multiplies whatever `grad_Y` last held by a zero diffusivity. The transport
# type is a type parameter of the solver, so the test adds no dispatch.
_species_gradients_skipped(solver) =
    _shared_species_diffusivity(solver) && _zero_molecular_diffusion(solver.transport)
_zero_molecular_diffusion(transport::ConstantTransport) = iszero(transport.mu0)
_zero_molecular_diffusion(::AbstractTransport) = false

function _species_gradients!(solver::SolverLike)
    decomp = solver.decomp
    for d in 1:3
        decomp.active[d] || continue
        for sp in 1:solver.equations.n_species
            deriv_scaled_along!(solver.grad_Y[d, sp], solver.Y[sp], solver, d, 1)
        end
    end
    return solver
end

# The shared-D_b species channels difference conserved components themselves
# (`:bulk` all of them, `:partial_density` the partial densities),
# ∂_d Q_c through the same scaled compact derivative, not a product-rule
# reconstruction from the stored gradients: the derivative is linear, and at
# uniform (u, p, T) every component is a fixed linear combination of the
# partial densities, so the fluxes of ρu and ρE are the same combinations of
# the species fluxes to round-off, which is what makes that state an exact
# discrete invariant of the term (reference/DESIGN.md, "The species
# channel"). n_cons or n_species line solves per active direction, on top of the
# 1 + n_species of `compute_rhs!`, whose n_species `_species_gradients_skipped`
# may remove. Every rank enters the solves. `tmp_a` is free
# at this point of the evaluation and holds the component being differenced.
# `level_smoothed` is true where the level-wide pass has already smoothed the
# sharpening flux's gradients (`_sharpening_fluxes!`).
function _bulk_gradients!(solver::SolverLike, Q, level_smoothed::Bool=false)
    decomp = solver.decomp
    n1f, n2f, n3f = padded_extent(decomp)
    # `:partial_density` differences the partial densities alone.
    n_diff = solver.art.species_flux === :bulk ? solver.equations.n_cons :
             solver.equations.n_species
    for c in 1:n_diff
        pointwise!(_copy_component_point!, solver.tmp_a, n1f, n2f, n3f,
                   solver.tmp_a, Q, c)
        for d in 1:3
            decomp.active[d] || continue
            deriv_scaled_along!(solver.grad_Q[d, c], solver.tmp_a, solver, d, 1)
        end
    end
    # A setup constant identical on every rank, so every rank enters the
    # sharpening flux's line solves or none does.
    _sharpening(solver) && _sharpening_fluxes!(solver, level_smoothed)
    return solver
end

# The interface sharpening flux of `ArtificialProperties.C_sharpen` (Brill,
# Olson & Bokman 2025, eq. 68), S_k = −ρ_k Γ g [ε ∇V_k − Σ_{j≠k} V_k V_j n̂_kj],
# written into the columns of `grad_Q` past the partial densities, which the
# partial-density channel leaves unused (`n_cons − n_species` = 4 of them per
# direction, which hold the 2(N − 1) gradients below for N ≤
# `SHARPEN_MAX_SPECIES`). The volume-fraction gradients of
# the first n_species − 1 species go there first, one line solve per species
# and direction, each followed by a copy that the artificial properties'
# smoother filters: the pair normals are built from the filtered gradients, as
# Brill, Olson & Bokman build theirs (eq. 69), so that a ringing tail, whose
# gradient changes sign from cell to cell, does not turn the compressive term
# into an alternating one. The diffusive term takes the unfiltered gradient.
# The pass then replaces the gradients point by point with S_k for every
# species, having read all of a point's values before writing any. The last
# species takes V_N = 1 − Σ_{k<N} V_k and ∇V_N = −Σ_{k<N} ∇V_k, so
# Σ_k ∇V_k = 0 holds pointwise and not only to the truncation error of the
# derivative. `tmp_a` is free once the partial densities are differenced, and
# `smooth!` exchanges halos, so every rank enters this or none does.
#
# On a tiled level with shared faces the smoothed gradients come from the
# level-wide pass instead (`level_smoothed`, `_level_sharpening_gradients!`),
# which smooths them across the faces as one patch spanning the tiles would;
# smoothed tile by tile, they take the closure rows of a field without ghosts
# at a shared face, and the gate and the normals then change along the tile
# lattice.
function _sharpening_fluxes!(solver::SolverLike, level_smoothed::Bool=false)
    decomp = solver.decomp
    N = solver.equations.n_species
    for sp in 1:(N - 1)
        _volume_fraction_gradients!(ntuple(d -> solver.grad_Q[d, N + sp], 3), solver,
                                    sp)
        # After every direction's derivative: `smooth!` takes `tmp_a` as its
        # scratch, and `tmp_a` holds the fraction being differenced.
        for d in 1:3
            decomp.active[d] || continue
            filtered = solver.grad_Q[d, 2N - 1 + sp]
            if level_smoothed
                copy_interior!(filtered, _sharpen_gradient(solver, solver.art, sp, d),
                               decomp)
            else
                copy_interior!(filtered, solver.grad_Q[d, N + sp], decomp)
                smooth!(filtered, solver)
            end
        end
    end
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    pointwise!(_sharpen_flux_point!, solver.rho, nx, ny, nz,
               FieldMatrix(solver.grad_Q), solver.eos, solver.field_tuples.Y,
               solver.rho, solver.c, solver.D_art[1], solver.inv_h[1],
               solver.inv_h[2], solver.inv_h[3], solver.h, decomp.active, N,
               _sharpening_constants(solver), o1, o2, o3)
    return solver
end

# The gradient of the volume fraction of species `sp` along every active
# dimension `d` into `out[d]`, through `tmp_a`, which holds the fraction on
# return. Collective over the patch's communicator.
function _volume_fraction_gradients!(out, solver::SolverLike, sp::Int)
    decomp = solver.decomp
    n1f, n2f, n3f = padded_extent(decomp)
    pointwise!(_volume_fraction_point!, solver.tmp_a, n1f, n2f, n3f,
               solver.tmp_a, solver.eos, solver.field_tuples.Y, sp,
               solver.equations.n_species)
    for d in 1:3
        decomp.active[d] || continue
        deriv_scaled_along!(out[d], solver.tmp_a, solver, d, 1)
    end
    return out
end

# The gate of the sharpening flux as a function of θ = ε_g/ℓ, ℓ the local
# logistic thickness of the pair fractions and ε_g = ε + D_b/Γ the thickness
# the flux and the channel's D_b hold together: 0 below the first value (ℓ
# above 3ε_g, a composition gradient resolved over more cells than an
# interface), 1 above the second (ℓ below 2ε_g), linear between. On a smooth
# profile D_b is small and ε_g is ε; at a shocked interface D_b is not, and
# measuring ℓ against ε alone closes the gate on the profile the two fluxes
# settle to. `SHARPEN_NORMAL` is the relative floor under the length of each
# pair normal.
const SHARPEN_GATE = (1 / 3, 1 / 2)
const SHARPEN_NORMAL = 1e-3
@inline function _sharpen_gate(θ::T) where {T}
    θ0, θ1 = T(SHARPEN_GATE[1]), T(SHARPEN_GATE[2])
    return clamp((θ - θ0) / (θ1 - θ0), zero(T), one(T))
end

@inline function _volume_fraction_point!(V, eos, Y, sp, n_species, i, j, k)
    @inbounds V[i, j, k] = _clipped_volume_fraction(eos, sp, Y, CartesianIndex(i, j, k),
                                                    n_species)
    return nothing
end

# The volume fractions, and the gradients along one direction from the
# `grad_Q` columns past `base`, as four-tuples, zero past the species count.
# The reads are conditional on the count, which is uniform over the launch.
@inline function _sharpen_fractions(eos, Y, I, N)
    T = eltype(Y[1])
    z = zero(T)
    x1 = _clipped_volume_fraction(eos, 1, Y, I, N)
    x2 = N > 2 ? _clipped_volume_fraction(eos, 2, Y, I, N) : z
    last = one(T) - (x1 + x2)
    return (x1, ifelse(N == 2, last, x2), ifelse(N == 3, last, z), z)
end

@inline function _sharpen_gradients(gQ, d, act, I, N, base)
    @inbounds begin
        T = eltype(gQ[1, 1])
        z = zero(T)
        g1 = act ? gQ[d, base + 1][I] : z
        g2 = act & (N > 2) ? gQ[d, base + 2][I] : z
        last = -(g1 + g2)
        return (g1, ifelse(N == 2, last, g2), ifelse(N == 3, last, z), z)
    end
end

@inline function _sharpen_flux_point!(gQ, eos, Y, rho, c, D_b, ih1, ih2, ih3, hh,
                                      act, n_species, sharp, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        T = eltype(rho)
        N = n_species
        C, width = sharp
        ih = (ih1, ih2, ih3)
        Δ = zero(T)
        for d in 1:3
            act[d] && (Δ = max(Δ, hh[d] / ih[d][I]))
        end
        ε = width * Δ
        Γ = C * c[I]
        tiny = eps(T)^2
        V = _sharpen_fractions(eos, Y, I, N)
        # Unfiltered gradients for the diffusive term, filtered ones for the
        # normals and the gate.
        G = (_sharpen_gradients(gQ, 1, act[1], I, N, N),
             _sharpen_gradients(gQ, 2, act[2], I, N, N),
             _sharpen_gradients(gQ, 3, act[3], I, N, N))
        F = (_sharpen_gradients(gQ, 1, act[1], I, N, 2N - 1),
             _sharpen_gradients(gQ, 2, act[2], I, N, 2N - 1),
             _sharpen_gradients(gQ, 3, act[3], I, N, 2N - 1))
        # The gate: ε_g times the pair-summed length of V_j ∇V_k − V_k ∇V_j,
        # which is V_k V_j ∇ln(V_k/V_j), over the summed |V_k V_j|. With two
        # species this is ε_g|∇V|/(V(1 − V)), the ratio ε_g/ℓ exactly on a
        # logistic of thickness ℓ.
        num = zero(T)
        den = zero(T)
        for a in 1:(N - 1), b in (a + 1):N
            num += sqrt(_pair_normal2(V, F, a, b))
            den += abs(V[a] * V[b])
        end
        ε_g = ε + D_b[I] / (Γ + tiny)
        θ = ε_g * num / (den + tiny)
        gate = _sharpen_gate(θ)
        Γg = Γ * gate
        ρ = rho[I]
        for d in 1:3
            act[d] || continue
            for a in 1:N
                # Σ_{b≠a} V_a V_b n̂_ab along d, with n̂_ab = m_ab/|m_ab|
                # regularized where the pair gradient vanishes. The floor is
                # symmetric in the pair, so n̂_ba = −n̂_ab exactly and the
                # compressive terms cancel in the sum over species.
                comp = zero(T)
                for b in 1:N
                    b == a && continue
                    vv = V[a] * V[b]
                    m = V[b] * F[d][a] - V[a] * F[d][b]
                    δ = (T(SHARPEN_NORMAL) * abs(vv) + tiny) / ε
                    comp += vv * m / sqrt(_pair_normal2(V, F, a, b) + δ^2)
                end
                ρa = ρ * _material_density_ratio(eos, a, Y, I, N)
                S = -ρa * Γg * (ε * G[d][a] - comp)
                gQ[d, N + a][I] = ifelse(Γg > zero(T), S, zero(T))
            end
        end
    end
    return nothing
end

# |V_b ∇V_a − V_a ∇V_b|², the squared length of the unnormalized pair normal.
@inline function _pair_normal2(V, G, a, b)
    s = zero(V[1])
    for d in 1:3
        m = V[b] * G[d][a] - V[a] * G[d][b]
        s += m * m
    end
    return s
end

"""
    refresh_primitives!(solver, Q)

Update the primitive fields on `solver` (`rho`, `u`, `v`, `w`, `p`, `T_ion`,
`c`, `cp_mix`, and `Y`) from `Q` by exchanging rank-boundary halos and calling
`primitives!`. This operation is collective and idempotent.

During `compute_rhs!`, including source terms and boundary corrections, the
primitive fields correspond to the state being evaluated. In a `run!` callback,
they instead correspond to the input state of the fifth RK stage and predate the
final stage update and subsequent [`filter_state!`](@ref) pass.

A callback must therefore call this function before reading the primitive
fields. `save_vtk`, `dissipation_rate`, and the mixing diagnostics perform this
update internally. The conserved state `Q` is current in a callback;
`mixture_density` and related functions read it without depending on the
conserved layout. A callback may also write to `Q`; `run!` enforces the
boundary conditions and refreshes the primitives from `Q` before the next step.
"""
refresh_primitives!(solver::SolverLike, Q) =
    (exchange_state!(Q, solver.decomp); primitives!(solver, Q); solver)

"""
    refresh_primitives!(solver, states::Vector)

The multi-patch form: every patch this rank holds, in patch order, from
the state vector aligned with `solver.patches`. Each patch's exchange is
collective over that patch's own communicator, whose ranks all hold it.
"""
function refresh_primitives!(solver::Solver, states::Vector{<:ConservedState})
    for (ps, Q) in eachpatch(solver, states)
        refresh_primitives!(ps, Q)
    end
    return solver
end

"""
    compute_primitives_and_gradients!(solver, Q, primitives_current=false)

Refresh halos, primitives, and the physical-component velocity gradients from
`Q`. Sharing this sequence between the RHS and the diagnostics gives both the
same parity and curvature-correction routing, and gradients taken from the
current state.

`solver.grad_u[d, j]` is overwritten with the physical component (∇u)_{dj},
zeroed on a collapsed dimension, and the metric curvature terms are added on
top. Every rank must call this function because each active dimension requires
a distributed line solve.

Pass `primitives_current = true` when [`refresh_primitives!`](@ref) has
run on this exact `Q` and only the gradients are wanted; the caller is then
responsible for the claim that nothing has touched `Q` since.

`arm = true`, which the right-hand side passes, first records from the
refreshed primitives which dimensions take the parent-level derivative mask
(`_arm_child_mask!`); a diagnostic leaves the record of the last right-hand
side in place and builds the mask itself from the primitives it refreshed.
"""
function compute_primitives_and_gradients!(solver::SolverLike, Q,
                                           primitives_current::Bool=false,
                                           arm::Bool=false)
    decomp = solver.decomp
    primitives_current || refresh_primitives!(solver, Q)
    arm ? _arm_child_mask!(solver) : _release_child_mask!(solver)
    vel = (solver.u, solver.v, solver.w)
    for jj in 1:3, d in 1:3
        if decomp.active[d]
            # scaled by 1/h_d incl. stretching Jacobian
            deriv_scaled_along!(solver.grad_u[d, jj], vel[jj], solver, d,
                                vel_parity(solver, d, jj))
        else
            fill!(solver.grad_u[d, jj], 0)
        end
    end
    metric_correct_gradients!(solver, solver.metric)   # additive curvature terms
    _mark_gradients!(solver)
    return solver
end

# The primitives and velocity gradients of `compute_rhs!`, kept where the
# coefficients were computed beforehand (`coefficients_current`): the
# level-wide pass (`_level_artificial!`) computed both from this state, the
# gradients into arrays of this patch's own (`_own_gradients`) or into its
# block of a stack's, and nothing has written either since. The derivative
# mask is armed afresh either way, since the pass armed each tile in turn.
function _gradient_step!(solver, Q, primitives_current::Bool,
                         coefficients_current::Bool)
    coefficients_current && return _arm_child_mask!(solver)
    return compute_primitives_and_gradients!(solver, Q, primitives_current, true)
end

# The artificial-property step of `compute_rhs!`: `compute_artificial!`, or,
# where the coefficients were computed beforehand (`coefficients_current`),
# only the compression switch of a gated β*, which reads this patch's velocity
# gradients and so cannot be applied before them, the shared workspace holding
# one patch's at a time. The branch sits here rather than in `compute_rhs!`,
# whose unoptimized body bench/audit.jl reads: there each branch statement
# counts as a value of type `Any`.
function _artificial_step!(solver, Q, coefficients_current::Bool)
    if coefficients_current
        _gated(solver.art) && gate_beta!(solver)
        return solver
    end
    return compute_artificial!(solver, Q)
end

"""
    compute_rhs!(solver, Q, dQ, primitives_current=false, coefficients_current=false)

Evaluate dQ/dt into the interior of `dQ` from the conserved state `Q`
(boundary conditions should be enforced on `Q` beforehand). Collapsed dimensions
contribute no derivatives; the axis dimension routes through parity-folded
plans with mirror-filled halos. The interior of `dQ` is zeroed first, so it is
overwritten, not accumulated into.

This is collective: it exchanges halos, runs a distributed line solve per active
dimension per field, and calls `correct_rhs!` for every boundary condition,
which under NSCBC carries collectives of its own. Every rank must call it at the
same point in the step.

The primitives, the gradients, the artificial coefficients, `strain_mag`,
`flux`, and the scratch fields `tmp_a`, `tmp_b`, `sensor` and `sensor_sp` on
`solver` are all overwritten; so are `pairbuf` and `pairout` wherever a paired
fold exists, and `ring_buf` under `detector = :d8`. Everything in that list
except the primitives, the artificial coefficients and the fold pair buffers
lives on the [`RHSWorkspace`](@ref) this patch shares with the rank's other
patches of the same extent, so on a multi-patch solver those fields carry the
patch evaluated last, not this one, once the call returns; the workspace
records that patch for `grad_u`, `strain_mag` and `sensor`. The docstring of
`compute_artificial!` records which of the sensor scratch fields are dead on
return and may therefore be borrowed by a later phase of the same call.

A trailing `primitives_current = true` skips the opening halo exchange and
primitives pass, and is valid only immediately after the caller performs both on
this same `Q`. See [`compute_primitives_and_gradients!`](@ref); [`step!`](@ref)
passes it for the first RK stage of a `prepared` step, where [`max_rate`](@ref)
has done the work. A further trailing `coefficients_current = true` keeps the
artificial coefficients the patch holds, with the primitives and the velocity
gradients they were computed from, applying only the compression switch of
`beta_sensor = :gated_strain` or `:dilatation`; a tiled level passes it after
computing the coefficients over the whole level (`_level_artificial!`), which
leaves each tile's gradients in arrays of its own. Both flags are positional, not
keywords, allowing `bench/audit.jl` to reach the body with `code_typed`, which
returns only the forwarding method of a function with keywords.
"""
function compute_rhs!(solver::SolverLike, Q, dQ, primitives_current::Bool=false,
                      coefficients_current::Bool=false)
    decomp = solver.decomp
    _reflux_zero!(solver)
    _gradient_step!(solver, Q, primitives_current, coefficients_current)
    _validate_transport_state!(solver, Q; current=true)
    _artificial_step!(solver, Q, coefficients_current)
    for d in 1:3
        decomp.active[d] || continue
        deriv_scaled_along!(solver.grad_T_ion[d], solver.T_ion, solver, d, 1)
    end
    _species_gradients_skipped(solver) || _species_gradients!(solver)
    # The shared-D_b species channels' gradients of conserved components; a
    # setup-constant branch identical on every rank, so no collective sits
    # below it unreached. Behind a function barrier so that the default
    # path's inferred body (bench/audit.jl) does not carry the branch.
    _shared_species_diffusivity(solver) &&
        _cold_bulk_gradients!(_cold(_root_part(solver)), _cold(_patch_part(solver)),
                              parent(Q), coefficients_current)
    assemble_fluxes!(solver, Q)
    # Physical wall fluxes must enter the compact divergence, including its
    # near-wall rows. All ranks visit the hooks in the same order; the wall
    # hooks only write locally owned planes and add no collectives.
    for d in 1:3, side in 1:2
        decomp.active[d] || continue
        correct_flux!(solver.bcs[d][side], solver, Q, d, side)
    end
    for d in 1:3
        exchange_dim_batch!(view(solver.flux, d, :), decomp, d)
    end
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    # On an unstretched Cartesian grid every scale factor is exactly 1, so
    # A_d ≡ 1 and inv_J ≡ 1: the A_d·F_d product below is a full-array copy
    # whose result equals its input, and the inv_J multiply is a no-op. Skipping
    # both removes three array streams per (component, dimension), 45 of them
    # for a 5-component 3-D RHS, in what the phase budget shows is the single
    # largest phase. Curved or stretched grids take the general path unchanged.
    unitgeom = solver.metric isa CartesianMetric && all(isnothing, solver.stretch)
    ghost = solver.interface_flux === :ghost
    _reflux_defer!(solver, true)
    for c in 1:solver.equations.n_cons
        pointwise!(_zero_component_point!, dQ, nx, ny, nz, dQ, c, o1, o2, o3)
        for d in 1:3
            decomp.active[d] || continue
            Fdc = solver.flux[d, c]
            σ = _flux_sign(solver, d, c)
            # Setup constants of the patch, identical on every rank of its
            # communicator, so each rank takes the same solves.
            if ghost && _interface_dim(solver, d)
                _cold_ghost_flux_divergence!(parent(dQ), c, Fdc,
                                             _cold(_root_part(solver)),
                                             _cold(_patch_part(solver)), parent(Q), d)
            elseif unitgeom
                div_subtract_along!(dQ, c, Fdc, solver, d, σ, nothing)
            else
                # tmp_b = A_d F_d over the full array; A_d is odd in r for the
                # cylindrical axis (A₁ = r), flipping the flux parity.
                Ad = solver.area_d[d]
                n1f, n2f, n3f = padded_extent(decomp)
                if _pressure_gradient(solver, d, c)
                    # inv_J ∂(A_d(F − p))/∂ξ_d + inv_h_d ∂p/∂ξ_d, p even
                    # across the axis, the origin and the poles (metric.jl
                    # says why).
                    pointwise!(_area_flux_less_point!, solver.tmp_b, n1f, n2f, n3f,
                               solver.tmp_b, Ad, Fdc, solver.p)
                    div_subtract_along!(dQ, c, solver.tmp_b, solver, d, σ,
                                        solver.inv_J)
                    pressure_subtract_along!(dQ, c, solver, d)
                else
                    pointwise!(_area_flux_point!, solver.tmp_b, n1f, n2f, n3f,
                               solver.tmp_b, Ad, Fdc)
                    div_subtract_along!(dQ, c, solver.tmp_b, solver, d, σ,
                                        solver.inv_J)
                end
            end
        end
        _reflux_component!(solver, dQ, c)
    end
    _reflux_defer!(solver, false)
    add_metric_sources!(solver, dQ, Q, solver.metric)
    for d in 1:3, side in 1:2
        decomp.active[d] || continue
        correct_rhs!(solver.bcs[d][side], solver, Q, dQ, d, side)
    end
    add_sources!(solver, dQ, Q, solver.tstage)
    # Returns `dQ`, so the release adds no statement of its own to the body
    # that bench/audit.jl reads.
    return _release_child_mask!(solver, dQ)
end

