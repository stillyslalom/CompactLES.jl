# Time integration: five-stage fourth-order low-storage Runge–Kutta
# (Carpenter & Kennedy 1994), CFL-limited timestep, and the outer run loop
# with conservative-variable filtering every `filter_interval` steps.

const RKA = (0.0,
             -567301805773.0 / 1357537059087.0,
             -2404267990393.0 / 2016746695238.0,
             -3550918686646.0 / 2091501179385.0,
             -1275806237668.0 / 842570457699.0)

const RKB = (1432997174477.0 / 9575080441755.0,
             5161836677717.0 / 13612068292357.0,
             1720146321549.0 / 2090206949498.0,
             3134564353537.0 / 4481467310338.0,
             2277821191437.0 / 14882151754819.0)

const RKC = (0.0,
             1432997174477.0 / 9575080441755.0,
             2526269341429.0 / 6820363962896.0,
             2006345519317.0 / 3224310063776.0,
             2802321613138.0 / 2924317926251.0)

"""
    predicted_dt(solver, control, rate) -> dt

Turn a measured CFL rate into the step to take: extrapolate the rate forward by
`control.predict` steps, then cap the growth against the previous step. `rate`
is the reduced rate from [`max_rate`](@ref); the result is `solver.cfl` divided
by the extrapolated rate, capped at `control.max_growth * solver.dt_prev`.

Each is skipped when its control is zero, which is the default for both, and
also while `solver.rate_prev` and `solver.dt_prev` are still zero, which holds
on a freshly built solver and after a rollback but not on a second `run!` of the
same solver. The extrapolation is one-sided and only ever raises the rate; see
the comment in the body. Nothing on the solver is modified here.
"""
function predicted_dt(solver::Solver, control::StepControl, rate)
    r = rate
    if control.predict > 0 && solver.rate_prev > 0
        # Linear extrapolation, one-sided: only ever raise the rate. A falling
        # rate means the flow is relaxing, and stepping out on that prediction
        # is how you overshoot the next shock.
        r = max(r, r + control.predict * (r - solver.rate_prev))
    end
    dt = solver.cfl / r
    if control.max_growth > 0 && solver.dt_prev > 0
        dt = min(dt, control.max_growth * solver.dt_prev)
    end
    return dt
end

"""
    Workspace(Q)
    Workspace(solver)

Reusable low-storage RK stage arrays. Pass a retained workspace to [`run!`](@ref)
or [`step!`](@ref) to avoid reallocating stage storage across calls.
"""
struct Workspace{A}
    dQ::A
    du::A
end

Workspace(Q::AbstractArray) = Workspace(zero(Q), zero(Q))
Workspace(states::Vector{<:ConservedState}) =
    Workspace([zero(Q) for Q in states], [zero(Q) for Q in states])
Workspace(solver::Solver) = Workspace(allocate_state(solver), allocate_state(solver))
"""
    step!(solver, Q, dQ, du, dt, prepared=false)
    step!(solver, Q, workspace, dt, prepared=false)

Advance the conserved state by one five-stage, fourth-order low-storage
Runge--Kutta step of size `dt`, returning `Q`. `dQ` and `du` are caller-provided
work arrays of the same shape as `Q`, holding the stage right-hand side and the
low-storage accumulator; both are overwritten during the step, as is `Q`. A
[`Workspace`](@ref) supplies the pair. `solver.tstage` is left at
`solver.t + dt`, and boundary conditions are enforced on `Q` at that time before
returning; `solver.t` itself is not advanced, which [`run!`](@ref) does.

Collective, since every stage evaluates [`compute_rhs!`](@ref).

`prepared = true` asserts that boundary conditions are already enforced on `Q` at
`solver.t` and that the primitive fields are current for it, which lets the first
stage skip a halo exchange and a primitives pass. [`run!`](@ref) can assert this
because it applies the boundary conditions itself immediately before calling
[`max_rate`](@ref), which performs the exchange and the primitives pass. A direct
caller should leave the default in place unless it has done the same. The
argument is positional for the reason given under [`compute_rhs!`](@ref).
"""
function step!(solver::Solver, Q, dQ, du, dt, prepared::Bool=false)
    decomp = solver.decomp
    for stage in 1:5
        solver.tstage = solver.t + oftype(solver.t, RKC[stage]) * dt
        # RKC[1] = 0, so a prepared caller's boundary values are the ones this
        # stage would compute; nothing between there and here has touched Q.
        first_prepared = prepared && stage == 1
        first_prepared || apply_bcs!(solver, Q)
        compute_rhs!(solver, Q, dQ, first_prepared)
        _rk_update!(decomp, solver.equations.n_cons, Q, dQ, du,
                    RKA[stage], RKB[stage], dt)
    end
    solver.tstage = solver.t + dt
    apply_bcs!(solver, Q)
    return Q
end

# The low-storage stage update over one patch's interior, shared between the
# single-patch and multi-patch step drivers.
@inline function _rk_point!(Q, dQ, du, A, B, dt, c, o1, o2, o3, i, j, k)
    @inbounds begin
        v = A * du[i+o1, j+o2, k+o3, c] + dt * dQ[i+o1, j+o2, k+o3, c]
        du[i+o1, j+o2, k+o3, c] = v
        Q[i+o1, j+o2, k+o3, c] += B * v
    end
    return nothing
end

function _rk_update!(decomp::Decomp, n_cons::Int, Q, dQ, du, A, B, dt)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    T = eltype(Q)
    At, Bt, dtt = T(A), T(B), T(dt)
    for c in 1:n_cons
        pointwise!(_rk_point!, Q, nx, ny, nz, Q, dQ, du,
                   At, Bt, dtt, c, o1, o2, o3)
    end
    return Q
end

"""
    step!(solver, states, dQs, dus, dt, prepared=false)

Multi-patch form of [`step!`](@ref): `states`, `dQs` and `dus` are vectors
aligned with `solver.patches`. Each stage evaluates every local patch's RHS and
update in the global patch order, then makes the interfaces consistent with
[`sync_patches!`](@ref) — the shared-plane averaging and ghost refill after
every stage. Collective over `solver.comm`. A solver built with
`subcycle = true` delegates to `subcycled_step!` instead.
"""
function step!(solver::Solver, states::Vector{<:ConservedState},
               dQs::Vector{<:ConservedState}, dus::Vector{<:ConservedState},
               dt, prepared::Bool=false)
    getfield(solver, :subcycle) &&
        return subcycled_step!(solver, states, dQs, dus, dt, prepared)
    patches = getfield(solver, :patches)
    for stage in 1:5
        solver.tstage = solver.t + oftype(solver.t, RKC[stage]) * dt
        first_prepared = prepared && stage == 1
        for (i, p) in enumerate(patches)
            ps = PatchSolver(solver, p)
            first_prepared || apply_bcs!(ps, states[i])
            compute_rhs!(ps, states[i], dQs[i], first_prepared)
        end
        for (i, p) in enumerate(patches)
            _rk_update!(p.decomp, solver.equations.n_cons, states[i], dQs[i],
                        dus[i], RKA[stage], RKB[stage], dt)
        end
        sync_patches!(solver, states)
        prolong_level_ghosts!(solver, states)
    end
    solver.tstage = solver.t + dt
    for (i, p) in enumerate(patches)
        apply_bcs!(PatchSolver(solver, p), states[i])
    end
    return states
end

step!(solver::Solver, Q, workspace::Workspace, dt, prepared::Bool=false) =
    step!(solver, Q, workspace.dQ, workspace.du, dt, prepared)

"""
    subcycled_step!(solver, states, dQs, dus, dt, prepared=false)

Berger–Oliger subcycled step (Stage 4 of `reference/AMR_GPU.md`): advance the
coarse level by one step of `dt` with the fine level frozen, then the fine
level by three steps of `dt/3`, its shell imposed at every fine stage time
from the cubic Hermite reconstruction of the coarse trajectory
([`hermite_level_shell!`](@ref)). The `t^{n+1}` Hermite endpoint costs one
extra coarse RHS evaluation per step, taken before the fine subcycles so it
samples the coarse trajectory, not the restricted composite.

The fine level filters its own state at its own step cadence — fine substep
`3·step + m` filters when `filter_interval` divides it — so each level
filters once per step of its own, exactly as an unrefined run does. Under a
positive `filter_cfl` the fine pass's relaxation weight reads the coarse
`dt · rate` product, an upper bound on the fine level's own CFL, so the fine
filter is never weaker than the convention intends. `run!`'s own per-step
filter pass covers the coarse level only in this mode, and the post-step
restriction then rebuilds the covered coarse region from the filtered fine
state.

Selected by `step!` when the solver was built with `subcycle = true`.
Collective, like the level coupling itself since G3c: the Hermite box saves
gather over the coarse communicator and every shell imposition carries the
component-distributed chain's ring Allgatherv, so every rank must take the
same substep sequence.
"""
function subcycled_step!(solver::Solver, states::Vector{<:ConservedState},
                         dQs::Vector{<:ConservedState},
                         dus::Vector{<:ConservedState}, dt,
                         prepared::Bool=false)
    lt = solver.level_transfer
    patches = getfield(solver, :patches)
    n_cons = solver.equations.n_cons
    coarse = patches[1]
    fine = patches[lt.fine_index]
    psc = PatchSolver(solver, coarse)
    psf = PatchSolver(solver, fine)
    Qc, dQc, duc = states[1], dQs[1], dus[1]
    Qf, dQf, duf = states[lt.fine_index], dQs[lt.fine_index], dus[lt.fine_index]
    t0 = solver.t
    # --- Coarse step over [t, t + dt]; the covered region is advanced too and
    # overwritten by the restriction afterwards, as in the global-dt mode.
    for stage in 1:5
        solver.tstage = t0 + oftype(t0, RKC[stage]) * dt
        first_prepared = prepared && stage == 1
        first_prepared || apply_bcs!(psc, Qc)
        compute_rhs!(psc, Qc, dQc, first_prepared)
        # RKC[1] = 0, so stage 1's dQ is the RHS at t^n on the unmodified Q.
        stage == 1 && save_level_box!(lt, coarse.decomp, Qc, dQc, false)
        _rk_update!(coarse.decomp, n_cons, Qc, dQc, duc,
                    RKA[stage], RKB[stage], dt)
    end
    solver.tstage = t0 + dt
    apply_bcs!(psc, Qc)
    compute_rhs!(psc, Qc, dQc, false)
    save_level_box!(lt, coarse.decomp, Qc, dQc, true)
    # --- Three fine steps of dt/3, boundary-forced from the Hermite box.
    dtf = dt / oftype(dt, 3)
    for m in 1:3
        tm = t0 + (m - 1) * dtf
        for stage in 1:5
            solver.tstage = tm + oftype(tm, RKC[stage]) * dtf
            θ = (oftype(dt, m - 1) + oftype(dt, RKC[stage])) / oftype(dt, 3)
            hermite_level_shell!(solver, states, θ, dt)
            apply_bcs!(psf, Qf)
            compute_rhs!(psf, Qf, dQf, false)
            _rk_update!(fine.decomp, n_cons, Qf, dQf, duf,
                        RKA[stage], RKB[stage], dtf)
        end
        solver.tstage = tm + dtf
        θ = oftype(dt, m) / oftype(dt, 3)
        hermite_level_shell!(solver, states, θ, dt)
        apply_bcs!(psf, Qf)
        if solver.filter_interval > 0 &&
           (3 * solver.step + m) % solver.filter_interval == 0
            filter_state!(psf, Qf)
            # The filter is not shell-preserving; re-impose the forcing so the
            # next substep (or the restriction) reads a consistent boundary.
            hermite_level_shell!(solver, states, θ, dt)
        end
    end
    solver.tstage = t0 + dt
    return states
end

"""
    compute_dt(solver, Q)

CFL-limited timestep from the acoustic (|u_d| + c) / (h_d · scalefactor_d)
rate and a diffusive rate built from the current (possibly artificial)
transport coefficients, reduced over all ranks. This is `solver.cfl` divided by
the rate [`max_rate`](@ref) returns, discarding the density it returns
alongside, so it carries that function's collective and its side effects on
`Q`'s halos and on the primitive fields.

The artificial coefficients are those left by the previous step, so the
diffusive rate lags by one. `StepControl.predict` extrapolates against that lag
and is off by default; the note at the top of `stepcontrol.jl` records the
measurement behind that default.
"""
compute_dt(solver::Solver, Q) = solver.cfl / max_rate(solver, Q)[1]
compute_dt(solver::Solver, states::Vector{<:ConservedState}) =
    solver.cfl / max_rate(solver, states)[1]

"""
    max_rate(solver, Q) -> (rate, rho_min)

Global maximum of the CFL rate, and the global minimum mixture density taken
directly from `Q`. Both quantities are evaluated in the same loop and reduced by
one `Allreduce`, so every rank must call this and all receive the same pair.

Before the loop it exchanges `Q`'s halos and refreshes the primitive fields from
`Q`, which [`run!`](@ref) relies on when it passes `prepared = true` to
[`step!`](@ref).

The direct density check is required because `primitives!` substitutes finite
placeholders where ρ ≤ 0; the resulting CFL rate can remain finite after the
state has lost positivity. Without this check, failure is not detected until
the diffusive term subsequently drives `dt` toward zero.
"""
function max_rate(solver::Solver, Q)
    exchange_state!(Q, solver.decomp)   # keep halos consistent for primitives!
    primitives!(solver, Q)
    rate, ρ_min = _local_max_rate(solver, Q)
    # One collective, not two: both quantities are reduced with `max` by
    # negating the density, and this runs every step of every run.
    red = MPI.Allreduce([rate, -ρ_min], max, solver.comm)
    return (red[1], -red[2])
end

"""
    max_rate(solver, states::Vector) -> (rate, rho_min)

Multi-patch form: the per-patch exchange, primitives pass and interior sweep
run patch by patch, and the two quantities reduce over `solver.comm` — the
whole rank set — exactly once, hoisted outside the patch loop as the
collective discipline requires.
"""
function max_rate(solver::Solver, states::Vector{<:ConservedState})
    T = typeof(solver.cfl)
    rate = zero(T)
    ρ_min = T(Inf)
    subcycle = getfield(solver, :subcycle)
    for (ps, Q) in eachpatch(solver, states)
        exchange_state!(Q, ps.decomp)
        primitives!(ps, Q)
        r, m = _local_max_rate(ps, Q)
        # A subcycled level ℓ advances at dt / 3^ℓ, so its rate constrains the
        # coarse step three times more weakly per level.
        subcycle && (r /= oftype(r, 3)^ps.patch.level)
        rate = max(rate, r)
        ρ_min = min(ρ_min, m)
    end
    red = MPI.Allreduce([rate, -ρ_min], max, solver.comm)
    return (red[1], -red[2])
end

# The interior sweep of max_rate over one patch, rank-local and free of
# collectives. `Array` storage keeps the fused serial loop; device storage
# (and FORCE_KA, which is how the test suite pins the launch path on one
# machine) evaluates the rate through a pointwise body into `tmp_a`/`tmp_b`
# and reduces each with the storage's own `maximum`/`minimum`. Both scratch
# fields are free here: max_rate runs before the step's first RHS evaluation.
# Maximum and minimum are exact and order-independent, so the two paths agree
# bitwise.
function _local_max_rate(solver::SolverLike, Q)
    if _cpu_storage(Q) && !FORCE_KA[]
        return _local_max_rate_loop(solver, Q)
    end
    return _local_max_rate_launch(solver, Q)
end

@inline function _rate_point!(rate_out, rhoq_out, Q, rho, u, v, w, carr, cparr,
                              mu_art, beta_art, kappa_art, D_art,
                              inv_h1, inv_h2, inv_h3, inv_r, cot_over_r,
                              metric, a1, a2, a3, h1, h2, h3,
                              mu0, Pr, Sc, n_species, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        T = eltype(rho)
        ρQ = zero(T)
        for sp in 1:n_species
            ρQ += Q[I, sp]
        end
        rhoq_out[I] = ρQ
        ρ = rho[I]
        ri = one(T) / ρ
        c = carr[I]
        cp = cparr[I]
        uv = (u[I], v[I], w[I])
        ih = (inv_h1, inv_h2, inv_h3)
        hh = (h1, h2, h3)
        act = (a1, a2, a3)
        acc = zero(T)
        dsum = zero(T)
        for d in 1:3
            act[d] || continue
            idx = ih[d][I] / hh[d]
            acc += (abs(uv[d]) + c) * idx
            dsum += idx * idx
        end
        acc += _curvature_rate_point(metric, a2, a3, inv_r, cot_over_r, I, uv)
        Dmax = mu0 / (Sc * ρ)
        for sp in 1:n_species
            Dmax = max(Dmax, mu0 / (Sc * ρ) + D_art[sp][I])
        end
        ν = (mu0 + mu_art[I] + beta_art[I]) * ri +
            (mu0 * cp / Pr + kappa_art[I]) * ri / cp + Dmax
        rate_out[I] = acc + 2 * ν * dsum
    end
    return nothing
end

function _local_max_rate_launch(solver::SolverLike, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    tr = solver.transport
    ft = solver.field_tuples
    pointwise!(_rate_point!, solver.tmp_a, nx, ny, nz,
               solver.tmp_a, solver.tmp_b, Q, solver.rho, solver.u, solver.v,
               solver.w, solver.c, solver.cp_mix, solver.mu_art,
               solver.beta_art, solver.kappa_art, ft.D_art,
               solver.inv_h[1], solver.inv_h[2], solver.inv_h[3],
               solver.inv_r, solver.cot_over_r, solver.metric,
               decomp.active[1], decomp.active[2], decomp.active[3],
               solver.h[1], solver.h[2], solver.h[3],
               tr.mu0, tr.Pr, tr.Sc, solver.equations.n_species, o1, o2, o3)
    rates = view(solver.tmp_a, o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz)
    rhos = view(solver.tmp_b, o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz)
    return (maximum(rates), minimum(rhos))
end

function _local_max_rate_loop(solver::SolverLike, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_species = solver.equations.n_species
    tr = solver.transport
    T = eltype(Q)
    rate = zero(T)
    ρ_min = T(Inf)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρQ = zero(T)
        for sp in 1:n_species
            ρQ += Q[I, sp]
        end
        ρ_min = min(ρ_min, ρQ)
        ρ = solver.rho[I]
        ri = one(T) / ρ
        c = solver.c[I]
        cp = solver.cp_mix[I]
        uv = (solver.u[I], solver.v[I], solver.w[I])
        acc = zero(T)
        dsum = zero(T)
        for d in 1:3
            decomp.active[d] || continue      # no resolved variation
            idx = solver.inv_h[d][I] / solver.h[d]      # inverse physical spacing
            acc += (abs(uv[d]) + c) * idx
            dsum += idx * idx
        end
        # Curvature-source stiffness. When an angular dimension is RESOLVED,
        # its source rate (|u_ang|/r) is smaller than its advective rate
        # (|u_ang|/(r Δang)) and is already covered above. When it is
        # COLLAPSED — axisymmetric-with-swirl being the important case — the
        # loop skips it entirely, yet ρu_θ²/r still drives u_r as a stiff
        # source at small r. That term is added here.
        acc += curvature_rate(solver, solver.metric, I, uv)
        Dmax = tr.mu0 / (tr.Sc * ρ)
        for sp in 1:solver.equations.n_species
            Dmax = max(Dmax, tr.mu0 / (tr.Sc * ρ) + solver.D_art[sp][I])
        end
        ν = (tr.mu0 + solver.mu_art[I] + solver.beta_art[I]) * ri +
            (tr.mu0 * cp / tr.Pr + solver.kappa_art[I]) * ri / cp + Dmax
        acc += 2 * ν * dsum
        rate = max(rate, acc)
    end
    return (rate, ρ_min)
end

"""
    curvature_rate(solver, metric, I, uv)

Rate contribution from geometric momentum sources on angular dimensions that
are collapsed (and therefore contribute no advective CFL term). Zero in
Cartesian coordinates and whenever the corresponding dimension is resolved.
"""
curvature_rate(solver, metric::Metric, I, uv) =
    _curvature_rate_point(metric, solver.decomp.active[2],
                          solver.decomp.active[3], solver.inv_r,
                          solver.cot_over_r, I, uv)

# The launchable form: plain arrays and activity flags in place of the solver,
# shared with the `_rate_point!` kernel body of `max_rate`.
@inline _curvature_rate_point(::CartesianMetric, a2, a3, inv_r, cot_over_r,
                              I, uv) = zero(first(uv))

@inline function _curvature_rate_point(::CylindricalMetric, a2, a3, inv_r,
                                       cot_over_r, I, uv)
    a2 && return zero(first(uv))            # covered by θ advection
    return @inbounds abs(uv[2]) * inv_r[I]  # ρu_θ²/r driving u_r
end

@inline function _curvature_rate_point(::SphericalMetric, a2, a3, inv_r,
                                       cot_over_r, I, uv)
    a = zero(first(uv))
    @inbounds begin
        a2 || (a += abs(uv[2]) * inv_r[I])
        a3 || (a += abs(uv[3]) * (inv_r[I] + abs(cot_over_r[I])))
    end
    return a
end

"""
    dt_report(solver, Q)

Diagnostic companion to [`compute_dt`](@ref), returning the NamedTuple
`(dt, rank, index, coords, dim, kind)`. `dt` is the same limited timestep
`compute_dt` returns and `rank` is the rank owning the point that set it; the
lowest such rank is named if several tie. `index` is that point's rank-local,
one-based interior index, `coords` its physical coordinates, `dim` the direction
carrying the largest acoustic rate there, and `kind` is `:acoustic`,
`:diffusive` or `:curvature`, whichever of the diffusive rate, the curvature rate
and the acoustic rate in direction `dim` is largest, ties going to `:acoustic`.
The acoustic entry in that comparison is the one direction, not the sum over
directions that the selecting rate and `dt` are built from.

Only `dt` and `rank` are global. The remaining four describe the calling rank's
own local maximum, so read them from the rank named by `rank`.

Collective: two `Allreduce`s, plus the halo exchange and primitives refresh that
[`max_rate`](@ref) also performs. Periodic evaluation distinguishes a physical
timestep restriction from one imposed by azimuthal spacing near a coordinate
singularity; see the CFL discussion in the README.
"""
function dt_report(solver::Solver, Q)
    _cpu_storage(Q) ||
        error("dt_report is a host sweep; copy the state to a CPU solver " *
              "for this diagnostic on a DeviceBackend")
    decomp = solver.decomp
    exchange_state!(Q, decomp)
    primitives!(solver, Q)
    o1, o2, o3 = decomp.n_halo_d
    tr = solver.transport
    best = (rate=-Inf, i=0, j=0, k=0, dim=0, kind=:none)
    @inbounds for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = solver.rho[I]; ri = 1 / ρ; c = solver.c[I]; cp = solver.cp_mix[I]
        uv = (solver.u[I], solver.v[I], solver.w[I])
        acc = 0.0; dsum = 0.0; wdim = 0; wrate = -Inf
        for d in 1:3
            decomp.active[d] || continue
            idx = solver.inv_h[d][I] / solver.h[d]
            rd = (abs(uv[d]) + c) * idx
            acc += rd; dsum += idx * idx
            rd > wrate && (wrate = rd; wdim = d)
        end
        crate = curvature_rate(solver, solver.metric, I, uv)
        Dmax = tr.mu0 / (tr.Sc * ρ)
        for sp in 1:solver.equations.n_species
            Dmax = max(Dmax, tr.mu0 / (tr.Sc * ρ) + solver.D_art[sp][I])
        end
        ν = (tr.mu0 + solver.mu_art[I] + solver.beta_art[I]) * ri +
            (tr.mu0 * cp / tr.Pr + solver.kappa_art[I]) * ri / cp + Dmax
        drate = 2 * ν * dsum
        total = acc + crate + drate
        if total > best.rate
            kind = drate > max(wrate, crate) ? :diffusive :
                   crate > wrate ? :curvature : :acoustic
            best = (rate=total, i=i, j=j, k=k, dim=wdim, kind=kind)
        end
    end
    grate = MPI.Allreduce(best.rate, max, solver.comm)
    mine = best.rate >= grate ? MPI.Comm_rank(solver.comm) : typemax(Int)
    owner = MPI.Allreduce(mine, min, solver.comm)
    return (dt=solver.cfl / grate, rank=owner, index=(best.i, best.j, best.k),
            coords=(xcoord(solver, 1, best.i), xcoord(solver, 2, best.j),
                    xcoord(solver, 3, best.k)),
            dim=best.dim, kind=best.kind)
end

"""
    positivity_floors(solver, Q, control) -> (rho_floor, e_floor)

Absolute floors for `apply_positivity_floor!`, derived from `Q` as
`control.floor_ratio` times the global minimum mixture density and the global
minimum internal energy over the interior.

Returns `(0, 0)`, which leaves the failsafe inactive, when `floor_ratio` is zero
and also when either reference is not itself positive: a state that has already
left the physical space supplies no scale to floor against, and [`run!`](@ref)
warns rather than inventing one.

Collective, one `Allreduce`. `run!` calls it once before its loop rather than
once per step, so a second `run!` on the same solver re-derives the floors from
the state that call starts with.
"""
function positivity_floors(solver::Solver, Q, control::StepControl)
    control.floor_ratio > 0 || return (0.0, 0.0)
    _cpu_storage(Q) ||
        error("StepControl.floor_ratio: the positivity failsafe is a host " *
              "sweep and is not yet supported on a DeviceBackend")
    ρ_min, e_min = _local_positivity_mins(solver, Q)
    red = MPI.Allreduce([ρ_min, e_min], min, solver.comm)
    (red[1] > 0 && red[2] > 0) || return (0.0, 0.0)
    return (control.floor_ratio * red[1], control.floor_ratio * red[2])
end

function positivity_floors(solver::Solver, states::Vector{<:ConservedState},
                           control::StepControl)
    control.floor_ratio > 0 || return (0.0, 0.0)
    ρ_min = Inf
    e_min = Inf
    for (ps, Q) in eachpatch(solver, states)
        r, e = _local_positivity_mins(ps, Q)
        ρ_min = min(ρ_min, r)
        e_min = min(e_min, e)
    end
    red = MPI.Allreduce([ρ_min, e_min], min, solver.comm)
    (red[1] > 0 && red[2] > 0) || return (0.0, 0.0)
    return (control.floor_ratio * red[1], control.floor_ratio * red[2])
end

function _local_positivity_mins(solver::SolverLike, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    ρ_min = Inf
    e_min = Inf
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = 0.0
        for sp in 1:n_species
            ρ += Q[I, sp]
        end
        ρ_min = min(ρ_min, ρ)
        # The internal energy is not recoverable where the density is not
        # positive. Such a point drives ρ_min below zero and disables the
        # failsafe on the line below, so skipping it here cannot hide anything.
        ρ > 0 || continue
        ke = 0.5 * (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / ρ
        e_min = min(e_min, (Q[I, i_energy] - ke) / ρ)
    end
    return (ρ_min, e_min)
end

"""
    apply_positivity_floor!(solver, Q, rho_floor, e_floor, scope) -> tally

Inspect the interior of `Q` for points outside the physical state space, repair
them in place as far as `scope` allows, and report the global
`(cells, low_energy, mass, energy, momentum)` tally of what that cost. The
floors come from `positivity_floors`, `scope` is `StepControl.floor_scope`, and
[`run!`](@ref) applies this after each completed step when
`StepControl.floor_ratio` is set.

Three repairs, in the order they have to run, since each depends on the state
the previous one leaves:

1. **Negative partial densities** are clipped to zero and the remaining positive
   ones rescaled onto the mixture density the point already carried, which
   leaves that density and therefore the mixture mass exactly unchanged.
2. **A mixture density below `rho_floor`** is raised to it, distributed over the
   positive partial densities, or onto the first species where there are none,
   which is the composition `primitives!` already substitutes at a point it
   cannot invert. Nothing conserves mass here, so the addition is tallied.
3. **Energy**, at a point whose internal energy `E/ρ − ½|u|²` is below
   `e_floor`. The point is counted as `low_energy` whatever the scope. It is
   repaired when `scope === :internal_energy`, and under `:representable` only
   when the *total* energy density `E` is itself below the floor, which is the
   state no frame can represent. The repair damps the velocity, scaling the
   momentum by `sqrt((E/ρ − e_floor) / (½|u|²))` with `E` untouched, so it moves
   kinetic energy into internal energy rather than creating any: total energy is
   conserved exactly and the momentum removed is tallied. Where there is no
   kinetic energy to convert, `E` being at or below the floor already, the
   fallback raises `E` instead and conserves the momentum exactly. Each branch
   conserves one of the two exactly and tallies what it did to the other.

`:representable` repairs only when `E` is below the floor, which is also the
condition selecting the fallback, so that scope always raises the energy and
never damps a velocity.

Collective, one `Allreduce` of the five-element tally, so every rank sees the
same totals and rank 0 reports on the whole domain rather than on its own block.

The repair writes the interior only. Halos are left as the step left them, and
nothing downstream depends on that: [`max_rate`](@ref) exchanges before reading
anything at the top of the next iteration.
"""
function apply_positivity_floor!(solver::Solver, Q, rho_floor, e_floor,
                                 scope::Symbol)
    tally = _local_positivity_repair!(solver, Q, rho_floor, e_floor, scope)
    red = MPI.Allreduce(collect(tally), +, solver.comm)
    return (cells=round(Int, red[1]), low_energy=round(Int, red[2]), mass=red[3],
            energy=red[4], momentum=red[5])
end

function apply_positivity_floor!(solver::Solver, states::Vector{<:ConservedState},
                                 rho_floor, e_floor, scope::Symbol)
    acc = (0.0, 0.0, 0.0, 0.0, 0.0)
    for (ps, Q) in eachpatch(solver, states)
        acc = acc .+ _local_positivity_repair!(ps, Q, rho_floor, e_floor, scope)
    end
    red = MPI.Allreduce(collect(acc), +, solver.comm)
    return (cells=round(Int, red[1]), low_energy=round(Int, red[2]), mass=red[3],
            energy=red[4], momentum=red[5])
end

function _local_positivity_repair!(solver::SolverLike, Q, rho_floor, e_floor,
                                   scope::Symbol)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    dV = cell_measure(solver)
    repair_e = scope === :internal_energy
    cells = 0.0; low_energy = 0.0; mass = 0.0; energy = 0.0; momentum = 0.0
    # Serial, as `max_rate` is: one pass per step over the same interior, and
    # only when a run enables the failsafe.
    @inbounds for k in 1:nz
        wk = quad_weight(solver, 3, k)
        for j in 1:ny
            wj = wk * quad_weight(solver, 2, j)
            for i in 1:nx
                I = CartesianIndex(i + o1, j + o2, k + o3)
                ρ = 0.0
                any_negative = false
                for sp in 1:n_species
                    q = Q[I, sp]
                    ρ += q
                    any_negative |= q < 0
                end
                if !any_negative && ρ >= rho_floor
                    ri = 1 / ρ
                    ke = 0.5 * (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) * ri
                    Q[I, i_energy] - ke >= ρ * e_floor && continue
                end
                # dV / inv_J is the physical cell volume, since inv_J carries any
                # stretching. This follows `volume_integral`'s convention, so a
                # tally is comparable with an integral of the field it perturbs.
                vol = wj * quad_weight(solver, 1, i) * dV / solver.inv_J[I]
                repaired = false
                if any_negative && ρ >= rho_floor
                    repaired = true
                    pos = 0.0
                    for sp in 1:n_species
                        pos += max(Q[I, sp], 0.0)
                    end
                    # pos >= ρ > 0, so the rescale only ever shrinks.
                    s = ρ / pos
                    for sp in 1:n_species
                        Q[I, sp] = max(Q[I, sp], 0.0) * s
                    end
                end
                if ρ < rho_floor
                    pos = 0.0
                    for sp in 1:n_species
                        pos += max(Q[I, sp], 0.0)
                    end
                    if pos > 0
                        s = rho_floor / pos
                        for sp in 1:n_species
                            Q[I, sp] = max(Q[I, sp], 0.0) * s
                        end
                    else
                        for sp in 1:n_species
                            Q[I, sp] = sp == 1 ? rho_floor : 0.0
                        end
                    end
                    mass += (rho_floor - ρ) * vol
                    ρ = rho_floor
                    repaired = true
                end
                ri = 1 / ρ
                ke = 0.5 * (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) * ri
                target = ρ * e_floor
                if Q[I, i_energy] - ke < target
                    low_energy += 1.0
                    if repair_e || Q[I, i_energy] < target
                        if Q[I, i_energy] > target && ke > 0
                            s = sqrt((Q[I, i_energy] - target) / ke)
                            pmag = sqrt(Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2)
                            Q[I, m1] *= s; Q[I, m2] *= s; Q[I, m3] *= s
                            momentum += (1 - s) * pmag * vol
                        else
                            energy += (target + ke - Q[I, i_energy]) * vol
                            Q[I, i_energy] = target + ke
                        end
                        repaired = true
                    end
                end
                repaired && (cells += 1.0)
            end
        end
    end
    return (cells, low_energy, mass, energy, momentum)
end

"""
    record_floor!(solver, tally) -> FloorTally

Accumulate one step's positivity-floor `tally` onto `solver.floor_tally` and
return it. Bookkeeping only. [`run!`](@ref) does the reporting, warning on the
first firing of a run and again in summary when that run ends, because a front
carrying a handful of repaired cells for the length of a run would otherwise
produce thousands of identical warnings.

The tally is cumulative across `run!` calls on the same solver, as the
wall-clock fields beside it are.
"""
function record_floor!(solver::Solver, tally)
    ft = solver.floor_tally
    ft.steps += 1
    ft.cells += tally.cells
    ft.low_energy += tally.low_energy
    ft.mass += tally.mass
    ft.energy += tally.energy
    ft.momentum += tally.momentum
    return ft
end

# Multi-patch counterparts of the per-step state operations `run!` composes.
# Each loops this rank's patches in the global order; none reduces.
apply_bcs!(solver::Solver, states::Vector{<:ConservedState}) =
    (foreach(((ps, Q),) -> apply_bcs!(ps, Q), eachpatch(solver, states)); states)

# Under subcycling the fine level filters itself inside `subcycled_step!` at
# its own step cadence, so the per-coarse-step pass here covers level 0 only.
function filter_state!(solver::Solver, states::Vector{<:ConservedState})
    subcycle = getfield(solver, :subcycle)
    for (ps, Q) in eachpatch(solver, states)
        subcycle && ps.patch.level > 0 && continue
        filter_state!(ps, Q)
    end
    return sync_patches!(solver, states)
end

# Zero the artificial coefficient arrays on every patch. Rollback support:
# the coefficients are regenerated by each RHS evaluation, so zero is the
# state a freshly built solver starts from, and the diffusive term of the
# retry's first rate measurement simply lags one step as it does on the first
# step of any run.
function _reset_artificial!(solver::Solver)
    for p in getfield(solver, :patches)
        fill!(p.mu_art, 0)
        fill!(p.beta_art, 0)
        fill!(p.kappa_art, 0)
        for D in p.D_art
            fill!(D, 0)
        end
    end
    return solver
end

_zero_state!(Q) = fill!(Q, 0)
_zero_state!(states::Vector{<:ConservedState}) =
    (foreach(Q -> fill!(Q, 0), states); states)
_reset_workspace!(w::Workspace) = (_zero_state!(w.dQ); _zero_state!(w.du); w)

# Savepoint copies for either state representation.
_snapshot(Q) = copy(Q)
_snapshot(states::Vector{<:ConservedState}) = [copy(Q) for Q in states]
_restore_state!(dst, src) = copyto!(dst, src)
function _restore_state!(dst::Vector{<:ConservedState}, src::Vector{<:ConservedState})
    for i in eachindex(dst)
        copyto!(dst[i], src[i])
    end
    return dst
end

# Interface and level consistency before the pre-step reads. The single-patch
# path has neither and skips this entirely.
_presync!(solver, Q) = Q
_presync!(solver, states::Vector{<:ConservedState}) =
    (sync_patches!(solver, states); sync_levels!(solver, states))

# Per-step level maintenance, after the state filter: restrict the fine state
# onto the covered coarse region, then re-impose the fine shell from the
# restricted coarse state. No-ops without refinement.
_post_step!(solver, Q) = Q
_post_step!(solver, states::Vector{<:ConservedState}) =
    sync_levels!(solver, states)

"""
    run!(solver, Q; tfinal, nmax=typemax(Int), callback=nothing, control=solver.control)
    run!(solver, Q, workspace; tfinal, ...)

Advance until `solver.t` reaches `tfinal` or `solver.step` reaches `nmax`,
filtering the conserved variables every `solver.filter_interval` steps and
invoking `callback` after each step. `nmax` bounds the solver's step counter
rather than the steps taken by this call, so a second `run!` on the same solver
must raise it. Returns `Q`, which is advanced in place.

The first form allocates a [`Workspace`](@ref) per call; pass `workspace` (as
the third positional argument or the keyword of the same name) to reuse one.

`callback` may be a bare `callback(solver, Q)` invoked every step (the original
contract, return value ignored), a [`Callback`](@ref) pairing a trigger with an
effect, or a tuple of those. A scheduled [`AtTime`](@ref) or [`EveryTime`](@ref)
trigger shortens `dt` over the preceding `control.landing_steps` steps so that a
step ends at the scheduled instant without an arbitrarily small final step. An
effect returning `true` ends the run after that step. `tfinal` uses a direct
clip, so scheduled callbacks do not alter the step sequence of a run that has
none.

`control` is a [`StepControl`](@ref) governing timestep prediction, the floors
below which the run is declared failed, and whether a failure is recoverable by
rolling back and lowering the CFL. It defaults to the one the solver was built
with. On an unrecoverable failure this throws [`SolverFailure`](@ref) rather
than continuing with a collapsed timestep; the note at the top of
`stepcontrol.jl` records what that failure mode looks like and which of the
three mechanisms was measured to help.

When `control.retries > 0`, the CFL is lowered in place on each retry.
Consequently, `solver.cfl` after a completed run records the value used to
complete the calculation and can be supplied to the next run's `Numerics`.

When `control.floor_ratio > 0`, `apply_positivity_floor!` inspects the conserved
state after each completed step and repairs it as far as `control.floor_scope`
allows, against floors `positivity_floors` derives once from the state this call
starts with. The first firing warns, the totals warn again when the run ends,
and `solver.floor_tally` carries them either way.

Each step updates `solver.t`, `solver.step`, `solver.dt_prev`,
`solver.rate_prev`, `solver.tstage`, `solver.wall_step` and
`solver.wall_total`; a rolled-back iteration records no wall time. The loop is
collective through [`max_rate`](@ref) and the line solves beneath
[`step!`](@ref), so every rank must call it with the same `tfinal`, `nmax` and
callbacks.
"""
function run!(solver::Solver, Q, workspace::Workspace;
              tfinal, nmax::Int=typemax(Int), callback=nothing,
              control::StepControl=solver.control)
    rank = MPI.Comm_rank(solver.comm)
    save = control.retries > 0 ? Savepoint(_snapshot(Q), solver.t, solver.step) : nothing
    attempts = 0
    dt_seen = 0.0
    rho_floor, e_floor = positivity_floors(solver, Q, control)
    if control.floor_ratio > 0 && rho_floor <= 0
        rank == 0 && @warn "run!: the positivity failsafe is inactive. The global " *
                           "minimum density or internal energy of the state " *
                           "entering this run is not positive, so floor_ratio " *
                           "has nothing to scale."
    end
    # Snapshot of the cumulative tally, so the summary below reports this
    # call's own repairs rather than everything the solver has accumulated.
    ft0 = solver.floor_tally
    floor_0 = (steps=ft0.steps, cells=ft0.cells, low_energy=ft0.low_energy,
               mass=ft0.mass, energy=ft0.energy, momentum=ft0.momentum)
    # Savepoints are suppressed at or below this step after a rollback. Without
    # it a retry re-saves its way forward to the state that failed and then
    # "rolls back" onto it, so every further retry starts from the corrupt state
    # and only the CFL moves. Observed as: rolled back to step 180, failed at
    # step 180, four times.
    guard_step = -1
    while solver.t < tfinal && solver.step < nmax
        # Timed from here rather than around step! alone: max_rate carries the
        # per-step Allreduce and the filter is a full set of line solves, so both
        # are step cost a user is trying to see. Callbacks are outside it — a
        # progress callback that reduces a diagnostic would otherwise be timing
        # itself, and reporting that as solver cost.
        wall_0 = time_ns()
        _maybe_regrid!(solver, Q, workspace, save)
        _presync!(solver, Q)
        # Boundary conditions before the rate measurement, for two reasons. The
        # step should be sized from the state it is about to advance, and the
        # previous iteration's filter_state! has smeared whatever the conditions
        # impose on the edge planes. With Q settled here, max_rate's halo exchange
        # and primitives pass also serve stage 1, which is what `prepared` below
        # asserts; that removes one sixth of both from the per-step cost.
        solver.tstage = solver.t
        apply_bcs!(solver, Q)
        rate, rho_min = max_rate(solver, Q)
        dt = predicted_dt(solver, control, rate)
        failure = check_step(control, dt, rho_min, dt_seen, solver.step,
                             solver.t, solver.cfl)
        if failure !== nothing
            (save === nothing || attempts >= control.retries) && throw(failure)
            attempts += 1
            _restore_state!(Q, save.Q)
            solver.t = save.t
            solver.step = save.step
            # Compounding: the CFL is reduced from its current value and never
            # restored from the savepoint, so three retries give backoff^3.
            solver.cfl *= control.cfl_backoff
            guard_step = failure.step
            # The rate history belongs to the abandoned trajectory; keeping it
            # would have the predictor extrapolate from states that no longer
            # exist. dt_seen goes too, or the relative floor would immediately
            # fire again against a dt from before the rollback.
            solver.dt_prev = zero(solver.dt_prev)
            solver.rate_prev = zero(solver.rate_prev)
            dt_seen = 0.0
            # A trajectory that failed NON-FINITE also leaves NaN in two
            # places a restore does not touch, either of which re-fails every
            # retry at the savepoint: the artificial coefficient arrays, which
            # max_rate reads before the retry's first RHS recomputes them, and
            # the low-storage accumulator, whose usual amnesia via RKA[1] = 0
            # cannot forget NaN (0.0 · NaN is NaN). Both were observed on a
            # subcycled Sod whose plain restart at the same lowered CFL
            # succeeded. The reset is deliberately gated on :nonfinite: after
            # a finite failure the stale coefficients are large where the
            # trouble is, and the small first dt they induce is a measured
            # part of the recovery the retry test pins.
            if failure.reason === :nonfinite
                _reset_artificial!(solver)
                _reset_workspace!(workspace)
            end
            # Instants between the savepoint and the failure were visited on a
            # trajectory that no longer exists. Re-arm them so the replacement
            # trajectory visits them too; see `rewind!` for what is and is not
            # rolled back.
            rewind_callbacks!(callback, save.t, save.step)
            rank == 0 && @warn "run!: $(failure.reason) at step $(failure.step); " *
                               "rolled back to step $(save.step) and lowered cfl to " *
                               "$(solver.cfl) (retry $attempts of $(control.retries))"
            continue
        end
        # Q has just passed its health check, which is the only moment it is
        # known good — the checks run on the state entering a step, so saving
        # after stepping would bank a state nothing has yet vetted.
        if save !== nothing && control.savepoint_interval > 0 &&
           solver.step > guard_step && solver.step % control.savepoint_interval == 0
            _restore_state!(save.Q, Q)
            save.t = solver.t
            save.step = solver.step
        end
        dt_seen = max(dt_seen, dt)
        # Clip to the endpoint AFTER the checks, and to the next scheduled
        # callback time as well: an AtTime trigger has to be landed on exactly,
        # not overshot, and nothing outside run! can reach dt to arrange that.
        # The gap is only applied when it is positive — a requested time already
        # behind solver.t would otherwise drive dt to zero or negative, which
        # stalls the run rather than firing anything.
        dt = min(dt, tfinal - solver.t)
        # An instant beyond `tfinal` is not reachable in this run, and aiming at
        # one makes the soft landing below halve the step against a target it
        # never arrives at. The case is not exotic: `EveryTime(0.003)` run to
        # `tfinal = 0.009` schedules its third instant at `0.006 + 0.003`, which
        # is one ULP ABOVE the 0.009 literal, so every step from there on had
        # `gap` a shade over `tfinal - solver.t` and the division by
        # `ceil(gap/dt) == 2` halved dt forever: 1.9e-13, 9.7e-14, 4.9e-14 and so
        # on for forty steps until `solver.t + dt` rounded onto `tfinal`. The run
        # did reach the endpoint and the trigger did fire — the whole cost was in
        # steps, which is why nothing caught it.
        #
        # Discarding the instant rather than clamping it to `tfinal` is what keeps
        # the endpoint out of the soft landing, per the note below. `tfinal` is
        # still clipped to, one line up; it is just never subdivided toward.
        next_instant = callback_next_time(callback, solver)
        gap = next_instant - solver.t
        # Soft landing. Clipping directly to the gap lands exactly but leaves an
        # arbitrarily small step before a scheduled instant: a dump every 1e-4
        # against a CFL step of 3.7e-5 gives steps of 3.7, 3.7, 3.7, 0.15 e-5.
        # That wastes a step and, with `max_growth` enabled, throttles the
        # several that follow. Dividing the remaining gap into `ceil(gap/dt)`
        # equal steps lands equally exactly with no step below dt/2.
        #
        # This applies to scheduled callback times only, not to `tfinal`. At the
        # endpoint no following step remains to be affected, and restricting the
        # change here leaves a run without scheduled callbacks stepping as it did
        # before, which is the condition the validation guards were measured
        # under.
        if next_instant <= tfinal && gap > 0 && gap < dt * control.landing_steps
            dt = gap / ceil(gap / dt)
        end
        prepared = true         # see the apply_bcs!/max_rate note above
        step!(solver, Q, workspace, dt, prepared)
        solver.t += dt
        solver.step += 1
        solver.dt_prev = dt
        solver.rate_prev = rate
        if solver.filter_interval > 0 && solver.step % solver.filter_interval == 0
            filter_state!(solver, Q)
        end
        _post_step!(solver, Q)
        # After the filter rather than immediately after step!, so that the state
        # entering the next iteration's checks is the repaired one whichever of
        # the two damaged it. The compact filter is not monotone, so it can
        # produce sub-floor values itself.
        if rho_floor > 0
            tally = apply_positivity_floor!(solver, Q, rho_floor, e_floor,
                                            control.floor_scope)
            if tally.cells > 0 || tally.low_energy > 0
                ft = record_floor!(solver, tally)
                ft.steps == floor_0.steps + 1 && rank == 0 &&
                    @warn "run!: the positivity failsafe saw $(tally.low_energy) " *
                          "cell(s) below the internal-energy floor and repaired " *
                          "$(tally.cells) at step $(solver.step), t = $(solver.t). " *
                          "Later steps are counted in solver.floor_tally and " *
                          "summarized when this run ends."
            end
        end
        # A rollback `continue`s above this, so an abandoned iteration never
        # records a step time — wall_total counts work that stood.
        solver.wall_step = (time_ns() - wall_0) / 1e9
        solver.wall_total += solver.wall_step
        run_callbacks!(callback, solver, Q) && break
    end
    ft = solver.floor_tally
    if ft.steps > floor_0.steps && rank == 0
        repaired = ft.cells - floor_0.cells
        @warn "run!: the positivity failsafe was active on " *
              "$(ft.steps - floor_0.steps) of this run's steps, over which it " *
              "saw $(ft.low_energy - floor_0.low_energy) cell(s) below the " *
              "internal-energy floor and repaired $repaired. Mass added " *
              "$(ft.mass - floor_0.mass), energy added " *
              "$(ft.energy - floor_0.energy), momentum removed " *
              "$(ft.momentum - floor_0.momentum)." *
              (repaired > 0 ? " A repaired cell makes this a repaired " *
                              "trajectory rather than a converged one." : "")
    end
    return Q
end

function run!(solver::Solver, Q; workspace=nothing, kwargs...)
    work = workspace === nothing ? Workspace(Q) : workspace
    return run!(solver, Q, work; kwargs...)
end

"""
    filter_weight(solver) -> w

Relaxation weight for one [`filter_state!`](@ref) pass, in `(0, 1]`.

`solver.filter_cfl == 0` returns `1`, which is the unrelaxed formulation: the
filter is applied at full strength on every pass, so it removes energy per
*application* rather than per unit time and its effective dissipation depends on
the timestep. Halving the CFL then doubles the number of applications covering
the same interval and doubles the dissipation, which is why the subgrid
dissipation does not converge as `dt → 0` at fixed resolution.

A positive `filter_cfl` restores that convergence by scaling the weight with the
step actually taken,

    w = filter_interval · dt · rate / filter_cfl

capped at 1. Since `compute_dt` sets `dt = cfl / rate`, the product `dt · rate`
recovers the CFL that was actually used, including `StepControl` backoff and the
shortening applied to land on a callback instant. So `w` is `cfl / filter_cfl`
in normal running, `filter_cfl` is the CFL at which one pass is applied at full
strength, and dissipation per unit time is invariant below it.

Reading `dt · rate` rather than `solver.cfl` makes a shortened step filter
proportionally less, which removes the truncated-final-step artifact in
`bench/tgv_energy.jl`.

The weight is also 1 whenever `dt_prev == 0`, since there is no step to scale
against. That holds on a freshly built solver and after a rollback; inside
`run!` the step is recorded before the filter runs, so the relaxed weight is in
force from the first pass.
"""
function filter_weight(solver::SolverLike{T}) where {T}
    solver.filter_cfl > 0 || return one(T)
    solver.dt_prev > 0 || return one(T)
    w = solver.filter_interval * solver.dt_prev * solver.rate_prev /
        solver.filter_cfl
    return min(one(T), w)
end

"""
    filter_state!(solver, Q)

Apply the compact filter to every conserved component of `Q` in place along every
active dimension, with batched per-dimension halo exchange and fold parity
routing from `cons_parity`: under the fold on the swept dimension each momentum
component takes the antipodal sign of its own velocity component, and partial
densities and energy are even. At the cylindrical axis this makes ρu_r and ρu_θ
odd and everything else even.

Collective, since each directional pass is a distributed line solve.
`solver.tmp_a` is scratch here. The filtered result lands in the interior of `Q`
only; each pass writes the halos from the exchange, so they are left
inconsistent with the filtered interior until the next step exchanges again.

Under a positive `filter_cfl` the result is relaxed toward the filtered state
rather than replaced by it, with the weight [`filter_weight`](@ref) supplies.
The weight is a reduced quantity by construction, since `rate_prev` comes from
the collective in `max_rate`, so every rank blends by the same amount without a
further reduction here.
"""
function filter_state!(solver::SolverLike, Q)
    decomp = solver.decomp
    comps = [view(Q, :, :, :, c) for c in 1:solver.equations.n_cons]
    w = filter_weight(solver)
    for d in 1:3
        decomp.active[d] || continue
        exchange_dim_batch!(comps, decomp, d)
        for c in 1:solver.equations.n_cons
            filt_along!(solver.tmp_a, comps[c], solver, d, cons_parity(solver, d, c))
            # w == 1 takes the original path exactly, so the default configuration
            # stays bit-identical to the unrelaxed solver.
            if w == 1
                copy_interior!(comps[c], solver.tmp_a, decomp)
            else
                blend_interior!(comps[c], solver.tmp_a, w, decomp)
            end
        end
    end
    return Q
end

# --- Top-level error reporting under many ranks.
#
# An uncaught exception is reported by *every* rank, so a SolverFailure at 448
# ranks produces 448 stacktraces — several thousand lines burying the one fact
# you need. And because the failures that matter here are raised off collective
# quantities (`max_rate` reduces, `check_step` reads the reduced result), they
# are usually identical on every rank, so 447 of those copies carry nothing.
#
# One backtrace, from rank 0, plus a one-line identification from anywhere else
# that failed — which is what catches the genuinely rank-local errors, like
# `plan_direction` rejecting one rank's block. Then MPI_Abort, so the ranks still
# blocked in a collective are killed outright rather than each unwinding through
# Julia's signal handler and printing a dump of its own.

"""
    mpi_main(body; comm = MPI.COMM_WORLD, exitcode = 1, grace = 0.5)

Run `body()` under a top-level error guard for large rank counts. Rank 0 prints a
full backtrace, other failing ranks print one line, and the function then calls
`MPI.Abort(comm, exitcode)`, which does not return. The return value on the
successful path is that of `body`. A failing rank other than rank 0 sleeps
`grace` seconds before aborting, so that rank 0's backtrace is not truncated by
the abort; see the comment in the body.

MPI drivers should use this guard to prevent an uncaught exception from
producing a full backtrace on every rank. For interrupts, pass
`--handle-signals=no` to `julia`; this function cannot intercept Julia's signal
handler while a rank is inside an MPI call.

Set `CL_ERROR_BACKTRACE=all` for a rank-local failure when rank 0 may remain
blocked in a collective and cannot print the primary backtrace.

```julia
mpi_main() do
    run!(solver, Q; tfinal = 1.0, callback = ProgressLog())
end
```
"""
function mpi_main(body; comm::MPI.Comm=MPI.COMM_WORLD, exitcode::Int=1,
                 grace::Float64=0.5)
    try
        return body()
    catch err
        rank = MPI.Comm_rank(comm)
        # One write, not several. Concurrent ranks writing a line in pieces
        # interleave character-by-character, and the result is unreadable well
        # before you get to 448 of them.
        # A failure on one rank alone leaves rank 0 blocked in a collective, so
        # nobody prints a backtrace and the one-liner carries no location. Set
        # CL_ERROR_BACKTRACE=all to get one from every failing rank — the right
        # setting for chasing a rank-local error, and the wrong one at 448 ranks
        # when they all fail together.
        all_bt = get(ENV, "CL_ERROR_BACKTRACE", "") == "all"
        if rank == 0 || all_bt
            print(stderr, "\nrank " * string(rank) * ": " *
                          sprint(showerror, err, catch_backtrace()) * "\n")
        else
            print(stderr, "rank " * string(rank) * ": " *
                          sprint(showerror, err) * "\n")
        end
        flush(stderr)
        # MPI_Abort kills the whole job, so whoever calls it first silences
        # everyone else — including rank 0 mid-backtrace, which is exactly the
        # output worth keeping. Non-zero ranks therefore yield briefly. If rank 0
        # did not fail it is blocked in a collective and gets killed regardless,
        # so the only cost is `grace` seconds on the way down.
        rank == 0 || sleep(grace)
        MPI.Abort(comm, exitcode)
    end
end
