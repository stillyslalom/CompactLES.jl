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

# --- Operator routing through folds ----------------------------------------

# The plans tuple is heterogeneous when a collapsed or folded dimension puts
# `nothing` in one of its slots. Indexing that tuple with a runtime `d` yields a
# union that the caller must split.
#
# The measured allocation was about 330 B per operator application and 11.9 kB
# per RHS on a planar (32, 16, 1) run, compared with 336 B per RHS in 3-D.
# Branching on `d` reduces this to 160 B per application and 3.8 kB per RHS.
#
# A concrete sentinel plan would remove the remaining union, but constructing
# one requires a `LineSolver` and communicator for a dimension that is never
# swept. The explicit branch avoids that unused state.
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
function deriv_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _plan_at(solver.deriv_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:deriv))
    end
    return out
end

"""
    div_along!(out, f, solver, d, σf)

Compact derivative of `f` along `d` through the divergence plans. These are
`solver.deriv_plans` except at a patch-interface end under
`interface_rhs = :extended`, where the gradient plans read exchanged ghost
data that a flux array does not carry, so the divergence keeps one-sided
closure rows there, the scheme's own or the cascade's for the neutral set
(`interface_divergence_closures`, `solver.div_plans`). A folded dimension draws on the
fold's own derivative plans instead, which is the same operator because folds
and patch interfaces never share a dimension. The flux-divergence loop and the
discrete-GCL construction `gcl_cotr!` go through here so the two apply the
identical operator. Same collective, halo, and fold contract as
[`deriv_along!`](@ref).
"""
function div_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _plan_at(solver.div_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:deriv))
    end
    return out
end

"""
    deriv_scaled_along!(out, f, solver, d, σf)

Compact derivative of `f` along active dimension `d`, scaled pointwise by
`solver.inv_h[d]` inside the scatter of the line solve. Interior results are
bit-identical to [`deriv_along!`](@ref) followed by `_scale_grad!`; only halo
cells of `out` differ (the two-pass rescale also scaled them, but they hold no
data any consumer reads without a fresh exchange). A fold dimension or a
device plan takes the two-pass route unchanged. Same collective, halo, and
fold contract as `deriv_along!`.
"""
function deriv_scaled_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    plan = _plan_at(solver.deriv_plans, d)
    if fold === nothing && !(plan isa DevicePlan)
        apply_along_scaled!(out, plan, f, solver.decomp, solver.inv_h[d])
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
subtraction pass. A fold dimension or a device plan takes that two-pass route
through `solver.tmp_a`. Same collective, halo, and fold contract as
`div_along!`.
"""
function div_subtract_along!(dQ, c::Int, f, solver::SolverLike, d::Int,
                             σf::Int, inv_J)
    fold = solver.folds[d]
    plan = _plan_at(solver.div_plans, d)
    if fold === nothing && !(plan isa DevicePlan)
        apply_along_subtract!(dQ, c, plan, f, solver.decomp, inv_J)
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
    return dQ
end

"""Compact filter of `f` along dimension `d` with antipodal sign `σf`.

Every rank in the directional sub-communicator must call this function. Its
halo and fold contract matches `deriv_along!`.
"""
function filt_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _plan_at(solver.filter_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:filter))
    end
    return out
end

"""
    smooth_along!(out, f, solver, d, σf)

Sensor smoother of `f` along dimension `d` with antipodal sign `σf`. This is
the Cook test filter, selected by `ArtParams.smoother`, and is a distinct
operator from `filt_along!`: the two coincide only under
`ArtParams(smoother = :compact)`, which aliases the filter plans and avoids
planning an operator of its own; the default `:gaussian` plans the explicit
nine-point stencil of [`gaussian_filter`](@ref). Only the artificial-property
sensors go through here, by way of `smooth!`.

A reflecting wall face is closed by the even rows of [`wall_closures`](@ref)
under `:gaussian`, since the fields smoothed here are even at a wall. The
`:compact` smoother keeps the state filter's plans and its own rows.

Every rank in the directional sub-communicator must call this function, as for
`deriv_along!`.
"""
function smooth_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _plan_at(solver.smooth_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:smooth))
    end
    return out
end

"""
    ring_along!(out, f, solver, d, σf, σw = 1)

Compact eighth derivative of `f` along dimension `d` with antipodal sign `σf`,
the ringing detector selected by `ArtParams(detector = :d8)`. Only `ring_sum!`
calls this, and only under that setting: `solver.ring_plans` is `nothing`
otherwise, which keeps this function off the default configuration's inference
path. Indexing that field under `:delta4` would throw. See `detect_sum!`.

`σw` is the field's sign across a reflecting wall on this dimension, and picks
the plan whose closure rows fold onto the node-centred mirror with that sign
([`wall_closures`](@ref)). Each dimension carries the two plans as a pair, so
the choice is a tuple index rather than a branch. Where neither face is such a
wall the pair holds one plan twice and the index is immaterial.

Every rank in the directional sub-communicator must call this function. Its
halo and fold contract matches `deriv_along!`.
"""
function ring_along!(out, f, solver::SolverLike, d::Int, σf::Int, σw::Int=1)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _wall_at(_plan_at(solver.ring_plans, d), σw), f,
                     solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:ring), σw)
    end
    return out
end

# The wall-sign half of a ring plan pair. The two plans differ only in their
# closure rows, so the selection is a tuple index and adds nothing to the hot
# path; see the comment above `_plan_at` for why the dimension is indexed the
# same way.
@inline _wall_at(pair, σw::Int) = σw > 0 ? pair[1] : pair[2]

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

# Antipodal signs of velocity and conserved components for the fold (if any)
# on dimension d; scalars, partial densities, and energy are always +1.
vel_parity(solver::SolverLike, d::Int, j::Int) =
    solver.folds[d] === nothing ? 1 : solver.folds[d].sigvel[j]
cons_parity(solver::SolverLike, d::Int, c::Int) =
    solver.folds[d] === nothing ? 1 :
    conserved_parity(solver.equations, solver.folds[d].sigvel, c)

assemble_fluxes!(solver::SolverLike, Q) =
    (_assemble_fluxes!(patch_fields(solver), solver.eos, solver.transport,
                       equation_layout(solver.equations),
                       _shared_species_diffusivity(solver), solver.art.species_flux,
                       Q); solver)

# No `::Type` argument here: a `Type` inside `pointwise!`'s Vararg defeats
# Julia's specialization heuristics and the body call turns into a per-point
# runtime dispatch, measured as assemble_fluxes! at 9× its cost. The element
# type comes off an array argument instead.
@inline function _fluxes_point!(Q, eos, rho, u, v, w, p, T_ion,
                                cp_mix, mu_art, beta_art, kappa_art, D_art, Y,
                                grad_u, gT, gY, flux, transport, n_species,
                                m1, m2, m3, i_energy, act, bulk, o1, o2, o3,
                                i, j, k)
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
        divu = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
        # T(2)/T(3), not the literal 2/3: the Float64 literal promotes the
        # normal stresses under a narrower T, making τ a heterogeneous tuple
        # whose runtime indexing is a dynamic field access, an InvalidIRError
        # on device. The Float64 value is identical.
        two_thirds = T(2) / T(3)
        τ11 = μ * (2*grad_u[1,1][I] - two_thirds * divu) + β * divu
        τ22 = μ * (2*grad_u[2,2][I] - two_thirds * divu) + β * divu
        τ33 = μ * (2*grad_u[3,3][I] - two_thirds * divu) + β * divu
        τ12 = μ * (grad_u[1,2][I] + grad_u[2,1][I])
        τ13 = μ * (grad_u[1,3][I] + grad_u[3,1][I])
        τ23 = μ * (grad_u[2,3][I] + grad_u[3,2][I])
        τ = ((τ11, τ12, τ13), (τ12, τ22, τ23), (τ13, τ23, τ33))
        # A collapsed dimension's flux is neither exchanged nor differenced,
        # so assembling it was a third of this phase wasted on a planar run.
        for d in 1:3
            act[d] || continue
            ud = uv[d]
            τd = τ[d]
            # Per-species diffusion with a correction velocity:
            # J_k = −ρ D_k ∇Y_k + ρ Y_k V_c, V_c = Σ_j D_j ∇Y_j,
            # which enforces Σ_k J_k = 0 exactly since ΣY_k = 1. Under the
            # shared-D_b species channels (`:partial_density`, `:bulk`) the
            # artificial part of this flux is added afterwards by
            # `_partial_density_flux_point!` or `_bulk_flux_point!`, so only
            # the molecular diffusivity D0 enters here; `D_art` then holds
            # D_b, which must not be read as a Fickian coefficient.
            Vc = zero(T)
            for sp in 1:n_species
                Dk = bulk ? molecular.D[sp] : molecular.D[sp] + D_art[sp][I]
                Vc += Dk * gY[d, sp][I]
            end
            hdiff = zero(T)              # Σ_k h_k J_{k,d}
            for sp in 1:n_species
                Dk = bulk ? molecular.D[sp] : molecular.D[sp] + D_art[sp][I]
                Jkd = ρ * (-Dk * gY[d, sp][I] + Y[sp][I] * Vc)
                flux[d, sp][I] = ρ * Y[sp][I] * ud + Jkd
                hdiff += species_enthalpy(eos, sp, point) * Jkd
            end
            flux[d, m1][I] = ρ * ud * uv[1] + (d == 1 ? pI : zero(T)) - τd[1]
            flux[d, m2][I] = ρ * ud * uv[2] + (d == 2 ? pI : zero(T)) - τd[2]
            flux[d, m3][I] = ρ * ud * uv[3] + (d == 3 ? pI : zero(T)) - τd[3]
            flux[d, i_energy][I] = (E + pI) * ud -
                           (uv[1]*τd[1] + uv[2]*τd[2] + uv[3]*τd[3]) -
                           κ * gT[d][I] + hdiff
        end
    end
    return nothing
end

# Keyed on the field storage, EOS, transport and `Q` only (see `PatchFields`),
# so every scheme, detector, dimensionality and patch wrapper shares one
# compiled body per (T, array type, EOS, transport).
function _assemble_fluxes!(f, eos, tr, eqi, shared::Bool, species_flux::Symbol, Q)
    n_species, n_cons, i_energy, (m1, m2, m3) = eqi
    decomp = f.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    ft = f.ft
    pointwise!(_fluxes_point!, f.rho, nx, ny, nz,
               Q, eos, f.rho, f.u, f.v, f.w, f.p,
               f.T_ion, f.cp_mix, f.mu_art, f.beta_art,
               f.kappa_art, ft.D_art, ft.Y, ft.grad_u,
               f.grad_T_ion, ft.grad_Y, ft.flux,
               tr, n_species, m1, m2, m3, i_energy,
               decomp.active, shared, o1, o2, o3)
    # The shared-D_b species channels, D_b read from `D_art[1]` (every
    # `D_art[k]` holds it) and ∂_d Q_c from the gradients `compute_rhs!` filled
    # into `grad_Q`: `:bulk` adds −D_b ∂_d Q_c to every component,
    # `:partial_density` the partial-density flux below. A separate pass rather
    # than a branch inside the body above, which is at the argument count
    # the launcher accepts. `species_flux` is a setup constant identical on
    # every rank, so the branch is safe with no collective below it.
    if shared && species_flux === :bulk
        pointwise!(_bulk_flux_point!, f.rho, nx, ny, nz,
                   ft.flux, f.D_art[1], FieldMatrix(f.grad_Q),
                   n_cons, decomp.active, o1, o2, o3)
    elseif shared && species_flux === :partial_density
        pointwise!(_partial_density_flux_point!, f.rho, nx, ny, nz,
                   ft.flux, eos, f.D_art[1], FieldMatrix(f.grad_Q),
                   f.u, f.v, f.w, f.T_ion, n_species,
                   m1, m2, m3, i_energy, decomp.active, o1, o2, o3)
    end
    return nothing
end

# The partial-density species channel of Brill, Olson & Bokman (2025, eqs.
# 38–40): J_k = −D_b ∂_d(ρY_k) on each species, the mass flux ΣJ carried into
# momentum as (ΣJ) u and into energy as (ΣJ) |u|²/2 + Σ_k e_k J_k, with the
# species internal energy e_k and not the enthalpy. At uniform (u, p, T) every
# added term is a fixed linear combination of the species fluxes, so the state
# is an exact discrete invariant, and no stress or conduction is added beyond
# what the mass flux carries.
@inline function _partial_density_flux_point!(flux, eos, D_b, gQ, u, v, w, T_ion,
                                              n_species, m1, m2, m3, i_energy,
                                              act, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        Db = D_b[I]
        uv = (u[I], v[I], w[I])
        point = _species_point(eos, T_ion[I])
        ke = (uv[1]^2 + uv[2]^2 + uv[3]^2) / 2
        for d in 1:3
            act[d] || continue
            Jsum = zero(Db)
            eJ = zero(Db)
            for sp in 1:n_species
                Jkd = -Db * gQ[d, sp][I]
                flux[d, sp][I] += Jkd
                Jsum += Jkd
                eJ += _species_internal_energy(eos, sp, point) * Jkd
            end
            flux[d, m1][I] += Jsum * uv[1]
            flux[d, m2][I] += Jsum * uv[2]
            flux[d, m3][I] += Jsum * uv[3]
            flux[d, i_energy][I] += Jsum * ke + eJ
        end
    end
    return nothing
end

# e_k(T) = h_k(T) − R_k T for the ideal-gas models; a single-component
# stiffened gas carries no composition gradient for the channel to act on.
@inline _species_internal_energy(eos, k::Int, T_ion) =
    species_enthalpy(eos, k, T_ion) - eos.Rk[k] * T_ion
@inline _species_internal_energy(eos::Union{StiffenedGas,StiffenedGasCoeffs},
                                 ::Int, T_ion) = eos.cv * T_ion
@inline _species_internal_energy(eos::Nasa9Mixture, k::Int, point::Nasa9Powers) =
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
# they are evaluated from the shell's gradient ring. The divergence is
# linear, so the ghost values' contribution is a second line solve over a
# field that is zero but for those ghosts, added to the first. What neither
# covers (the artificial fluxes and the wall corrections) keeps the one-sided
# rows of `div_plans`. Without molecular transport the split is the inviscid
# one alone. The split costs one pointwise pass and, where a remainder
# exists, a second pass and a second line solve per component and interface
# dimension; the molecular part adds one pass, one halo exchange and one
# line solve per component. It allocates nothing beyond `tmp_b` and, on a
# device plan, `tmp_a`, and the `ghost_flux` arrays sized at construction.

"Whether dimension `d` of this patch has an interface end the divergence closes."
@inline _interface_dim(solver::SolverLike, d::Int) =
    _plan_at(solver.div_plans, d) !== _plan_at(solver.deriv_plans, d)

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
    return !(per || bc_lo isa InterfaceBC) || !(per || bc_hi isa InterfaceBC)
end

# The ghost-differenced flux of component `c` along `d` at every padded point
# into `out`: the inviscid flux, plus the molecular flux `G` under `viscous`,
# or, under `remainder`, the assembled flux `F` less it. The inviscid
# expressions are those of `_fluxes_point!`, so on an inviscid run the
# remainder is exactly zero. At an interface end `F`'s ghosts hold no data
# and neither does the remainder there; the divergence rows that take it
# read none, and `G` is zero there (`_molecular_flux_point!`).
@inline function _inviscid_flux_point!(out, F, Q, rho, u, v, w, p, Y, G, c, d,
                                       n_species, m1, m2, m3, i_energy,
                                       remainder, viscous, i, j, k)
    T = eltype(rho)
    @inbounds begin
        I = CartesianIndex(i, j, k)
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
        out[I] = remainder ? F[I] - f : f
    end
    return nothing
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

# The molecular flux along `d` over the padded block from the interior
# gradients: evaluated where the point is interior along every active
# dimension, zero elsewhere, so an interface end's ghost layers start from
# zero and the rank halos along `d` take the neighbour's values in the
# exchange that follows. `stride` is the padded extent along the third
# dimension, which a stacked launch adds to `k` once per tile.
@inline function _molecular_flux_point!(G, eos, rho, u, v, w, T_ion, cp_mix, Y,
                                        grad_u, gT, gY, transport, n_species,
                                        m1, m2, m3, i_energy, d, n_cons, nl, pad,
                                        stride, i, j, k)
    T = eltype(rho)
    @inbounds begin
        I = CartesianIndex(i, j, k)
        kl = (k - 1) % stride + 1
        inside = pad[1] < i <= pad[1] + nl[1] && pad[2] < j <= pad[2] + nl[2] &&
                 pad[3] < kl <= pad[3] + nl[3]
        if !inside
            for c in 1:n_cons
                G[I, c] = zero(T)
            end
            return nothing
        end
        molecular = transport_at(transport, eos, T_ion, rho, cp_mix, Y, I)
        gu = ((grad_u[1, 1][I], grad_u[1, 2][I], grad_u[1, 3][I]),
              (grad_u[2, 1][I], grad_u[2, 2][I], grad_u[2, 3][I]),
              (grad_u[3, 1][I], grad_u[3, 2][I], grad_u[3, 3][I]))
        _molecular_flux!(G, I, eos, rho[I], (u[I], v[I], w[I]), T_ion[I], Y,
                         molecular, gu, gT[d][I], sp -> gY[d, sp][I], n_species,
                         m1, m2, m3, i_energy, d)
    end
    return nothing
end

# Phase one of the molecular ghost flux along an interface dimension `d`: the
# interior molecular flux into the patch's `ghost_flux[d]`, with its rank
# halos along `d` exchanged, from the gradients this evaluation computed.
# Collective over the patch's communicator.
function _molecular_ghost_flux!(solver::SolverLike, d::Int)
    decomp = solver.decomp
    eq = solver.equations
    m1, m2, m3 = eq.i_mom
    n1f, n2f, n3f = padded_extent(decomp)
    ft = solver.field_tuples
    G = solver.ghost_flux[d]
    pointwise!(_molecular_flux_point!, G, n1f, n2f, n3f,
               G, solver.eos, solver.rho, solver.u, solver.v, solver.w,
               solver.T_ion, solver.cp_mix, ft.Y, ft.grad_u, solver.grad_T_ion,
               ft.grad_Y, solver.transport, eq.n_species, m1, m2, m3,
               eq.i_energy, d, eq.n_cons, decomp.n_local, decomp.n_halo_d, n3f)
    exchange_dim_batch!(ComponentViews(G), decomp, d)
    return solver
end

# dQ[:, c] -= D_div(F - P) + D_ext(P) along `d`, P the ghost-differenced
# flux: the first through the divergence plans, skipped where the remainder
# is identically zero, the second through the gradient plans, reading the
# ghost fluxes. Both are collective line solves along `d`, and the branch
# before the first is a setup constant of the patch. `F` itself is left as
# assembled. The first component's call fills the molecular flux of every
# component (`_molecular_ghost_flux!`), which the conserved-component loop of
# `compute_rhs!` reaches before any other along `d`.
function _ghost_flux_divergence!(dQ, c::Int, Fdc, solver::SolverLike, Q, d::Int)
    decomp = solver.decomp
    eq = solver.equations
    m1, m2, m3 = eq.i_mom
    n1f, n2f, n3f = padded_extent(decomp)
    viscous = _ghost_viscous(solver)
    viscous && c == 1 && _molecular_ghost_flux!(solver, d)
    G = solver.ghost_flux[d]
    if _flux_remainder(solver, d)
        pointwise!(_inviscid_flux_point!, solver.tmp_b, n1f, n2f, n3f,
                   solver.tmp_b, Fdc, Q, solver.rho, solver.u, solver.v, solver.w,
                   solver.p, solver.field_tuples.Y, G, c, d, eq.n_species, m1, m2, m3,
                   eq.i_energy, true, viscous)
        div_subtract_along!(dQ, c, solver.tmp_b, solver, d, 1, nothing)
    end
    pointwise!(_inviscid_flux_point!, solver.tmp_b, n1f, n2f, n3f,
               solver.tmp_b, Fdc, Q, solver.rho, solver.u, solver.v, solver.w,
               solver.p, solver.field_tuples.Y, G, c, d, eq.n_species, m1, m2, m3,
               eq.i_energy, false, viscous)
    _ext_subtract_along!(dQ, c, solver.tmp_b, solver, d)
    return dQ
end

# dQ[:, c] -= D_ext(f) along `d` through the gradient plans, whose interface
# rows read `f`'s ghost layers; a device plan takes the two-pass route
# through `tmp_a`.
function _ext_subtract_along!(dQ, c::Int, f, solver::SolverLike, d::Int)
    decomp = solver.decomp
    plan = _plan_at(solver.deriv_plans, d)
    if plan isa DevicePlan
        apply_along!(solver.tmp_a, plan, f, decomp)
        nx, ny, nz = decomp.n_local
        o1, o2, o3 = decomp.n_halo_d
        pointwise!(_subtract_div_point!, dQ, nx, ny, nz,
                   dQ, solver.tmp_a, c, o1, o2, o3)
    else
        apply_along_subtract!(dQ, c, plan, f, decomp, nothing)
    end
    return dQ
end

# --- Phase two: the molecular flux through interface ends ---------------------
#
# Run by `_level_rhs!` after every patch of a level has evaluated its
# right-hand side, so each patch's `ghost_flux` holds its interior molecular
# flux. The level's flux records copy each same-level neighbour's interior
# values into the abutting ghost layers, over the same records that refill the
# state's ghosts; each coarse-fine face's ghost layers are evaluated from the
# shell's gradient ring. The ghost values alone then go through the gradient
# plans and are subtracted from `dQ`: the divergence is linear, so this
# completes the phase-one solve, whose ghosts were zero.

# Whether the patch's face `side` of `d` is an interface end whose ghost
# layers this rank holds: a same-level or coarse-fine face (both are
# `InterfaceBC`s), at the patch's own edge of the decomposition.
@inline function _ghost_face(solver::SolverLike, d::Int, side::Int)
    bc = solver.bcs[d][side]
    bc isa InterfaceBC || return false
    decomp = solver.decomp
    return side == 1 ? at_lo_edge(decomp, d) : at_hi_edge(decomp, d)
end

# `out` ← component `c` of `G` on the ghost layers of the interface ends along
# `d` flagged in `ends` (over the interior transverse range), zero elsewhere.
@inline function _ghost_only_point!(out, G, c, d, ends, nl, pad, i, j, k)
    T = eltype(out)
    @inbounds begin
        I = CartesianIndex(i, j, k)
        idx = (i, j, k)
        v = zero(T)
        transverse = true
        for e in 1:3
            e == d && continue
            transverse &= pad[e] < idx[e] <= pad[e] + nl[e]
        end
        if transverse && ((ends[1] && idx[d] <= pad[d]) ||
                          (ends[2] && idx[d] > pad[d] + nl[d]))
            v = G[I, c]
        end
        out[I] = v
    end
    return nothing
end

# The molecular flux along `d` on one coarse-fine face's ghost layers, from
# the patch's state and primitives there (the imposed shell) and the gradient
# ring: the primitive gradients follow from the conserved ones by the chain
# rule, ∂u = (∂(ρu) − u ∂ρ)/ρ, ∂Y_k = (∂(ρY_k) − Y_k ∂ρ)/ρ, and ∂e from ∂E,
# with ∂T from ∂e and the ∂Y_k through `_temperature_gradient`. The launch
# box is (ghost layers along `d`) × (interior transverse), `base` the padded
# index before the first layer.
@inline function _coarse_fine_flux_point!(G, Q, eos, rho, u, v, w, p, T_ion, cp_mix,
                                          Y, gring, table, off, pad, transport,
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
        gu = ntuple(j -> ntuple(m -> (gring[at, 3 * (mom[m] - 1) + j] -
                                      uv[m] * drho[j]) / ρ, 3), 3)
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
        ps = PatchSolver(solver, patches[pi])
        lt = lev.index == 0 ? nothing : lev.transfers[lev.tiles[k]]
        _patch_ghost_fluxes!(ps, lt, states[pi], dQs[pi], n_cons)
    end
    return dQs
end

function _patch_ghost_fluxes!(solver::SolverLike, lt, Q, dQ, n_cons::Int)
    decomp = solver.decomp
    eq = solver.equations
    m1, m2, m3 = eq.i_mom
    n1f, n2f, n3f = padded_extent(decomp)
    pad = decomp.n_halo_d
    nl = decomp.n_local
    ft = solver.field_tuples
    for d in 1:3
        G = solver.ghost_flux[d]
        size(G, 4) > 0 || continue
        # The coarse-fine faces' ghost layers, from the gradient ring. Every
        # rank of the patch takes the same branches: the conditions are the
        # patch's, and `_ghost_face` only restricts the writes to the edge.
        if lt !== nothing && lt.gradients !== nothing
            gring = lt.gradients.gring
            if _device_path(G)
                dg = similar(parent(G), size(gring))
                copyto!(dg, gring)
                gring = dg
            end
            for side in 1:2
                parent_fed(solver.bcs[d][side]) && _ghost_face(solver, d, side) ||
                    continue
                o1, o2 = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
                base = side == 1 ? 0 : pad[d] + nl[d]
                pointwise!(_coarse_fine_flux_point!, G, pad[d], nl[o1], nl[o2],
                           G, Q, solver.eos, solver.rho,
                           solver.u, solver.v, solver.w, solver.p, solver.T_ion,
                           solver.cp_mix, ft.Y, gring, (lt.shell.table...,),
                           decomp.offset, pad, solver.transport, eq.n_species,
                           m1, m2, m3, eq.i_energy, d, base)
            end
        end
        ends = (_ghost_face(solver, d, 1), _ghost_face(solver, d, 2))
        for c in 1:n_cons
            pointwise!(_ghost_only_point!, solver.tmp_b, n1f, n2f, n3f,
                       solver.tmp_b, G, c, d, ends, nl, pad)
            _ext_subtract_along!(dQ, c, solver.tmp_b, solver, d)
        end
    end
    return dQ
end

# Entries for the `_cold` calls in `compute_rhs!`. They take the arrays under
# the `ConservedState` wrappers, which are already on the heap, and rewrap
# them here: the immutable wrapper itself would be boxed crossing the dynamic
# call, 16 B per argument per call.
_cold_bulk_gradients!(solver, q) = _bulk_gradients!(solver, ConservedState(q))
_cold_ghost_flux_divergence!(dq, c::Int, Fdc, solver, q, d::Int) =
    _ghost_flux_divergence!(ConservedState(dq), c, Fdc, solver, ConservedState(q), d)

@inline function _copy_component_point!(dest, Q, c, i, j, k)
    @inbounds dest[i, j, k] = Q[i, j, k, c]
    return nothing
end

# The mass-fraction gradients `grad_Y`. Two terms read them: the molecular part
# of the species flux, which multiplies them by the molecular diffusivity, and
# the transverse terms of `NSCBCInflowBC`. Under a shared-D_b species channel
# with `Transport(mu0 = 0)` the first is identically zero, so `compute_rhs!`
# skips the n_species line solves per direction and the inflow condition takes
# them itself (`correct_rhs!`, above its early return). The flux body then
# multiplies whatever `grad_Y` last held by a zero diffusivity. The transport
# type is a type parameter of the solver, so the test adds no dispatch.
_species_gradients_skipped(solver) =
    _shared_species_diffusivity(solver) && _zero_molecular_diffusion(solver.transport)
_zero_molecular_diffusion(transport::Transport) = iszero(transport.mu0)
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
function _bulk_gradients!(solver::SolverLike, Q)
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
    return solver
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
"""
function compute_primitives_and_gradients!(solver::SolverLike, Q,
                                           primitives_current::Bool=false)
    decomp = solver.decomp
    primitives_current || refresh_primitives!(solver, Q)
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

"""
    compute_rhs!(solver, Q, dQ, primitives_current=false)

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
has done the work. It is positional, not a keyword, allowing `bench/audit.jl`
to reach the body with `code_typed`, which returns only the
forwarding method of a function with keywords.
"""
function compute_rhs!(solver::SolverLike, Q, dQ, primitives_current::Bool=false)
    decomp = solver.decomp
    compute_primitives_and_gradients!(solver, Q, primitives_current)
    _validate_transport_state!(solver, Q; current=true)
    compute_artificial!(solver, Q)
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
        _cold_bulk_gradients!(_cold(solver), parent(Q))
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
    for c in 1:solver.equations.n_cons
        pointwise!(_zero_component_point!, dQ, nx, ny, nz, dQ, c, o1, o2, o3)
        for d in 1:3
            decomp.active[d] || continue
            Fdc = solver.flux[d, c]
            σ = solver.folds[d] === nothing ? 1 : solver.folds[d].sigflux[c]
            # Setup constants of the patch, identical on every rank of its
            # communicator, so each rank takes the same solves.
            if ghost && _interface_dim(solver, d)
                _cold_ghost_flux_divergence!(parent(dQ), c, Fdc, _cold(solver),
                                             parent(Q), d)
            elseif unitgeom
                div_subtract_along!(dQ, c, Fdc, solver, d, σ, nothing)
            else
                # tmp_b = A_d F_d over the full array; A_d is odd in r for the
                # cylindrical axis (A₁ = r), flipping the flux parity.
                Ad = solver.area_d[d]
                n1f, n2f, n3f = padded_extent(decomp)
                pointwise!(_area_flux_point!, solver.tmp_b, n1f, n2f, n3f,
                           solver.tmp_b, Ad, Fdc)
                div_subtract_along!(dQ, c, solver.tmp_b, solver, d, σ,
                                    solver.inv_J)
            end
        end
    end
    add_metric_sources!(solver, dQ, Q, solver.metric)
    for d in 1:3, side in 1:2
        decomp.active[d] || continue
        correct_rhs!(solver.bcs[d][side], solver, Q, dQ, d, side)
    end
    add_sources!(solver, dQ, Q, solver.tstage)
    return dQ
end

