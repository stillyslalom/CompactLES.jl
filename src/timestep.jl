# Time integration: five-stage fourth-order low-storage Runge–Kutta
# (Carpenter & Kennedy 1994), CFL-limited timestep, and the outer run loop
# with per-step conservative-variable filtering.

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
    step!(solver, Q, dQ, du, dt)

Advance the conserved state by one RK45 step of size `dt`. `dQ` and `du` are
caller-provided work arrays of the same shape as `Q`.
"""
function step!(solver::Solver, Q, dQ, du, dt)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    for stage in 1:5
        solver.tstage = solver.t + RKC[stage] * dt
        apply_bcs!(solver, Q)
        compute_rhs!(solver, Q, dQ)
        A = RKA[stage]
        B = RKB[stage]
        for c in 1:solver.n_cons
            @threaded nx*ny*nz for k in 1:nz
                @inbounds for j in 1:ny, i in 1:nx
                    v = A * du[i+o1, j+o2, k+o3, c] + dt * dQ[i+o1, j+o2, k+o3, c]
                    du[i+o1, j+o2, k+o3, c] = v
                    Q[i+o1, j+o2, k+o3, c] += B * v
                end
            end
        end
    end
    solver.tstage = solver.t + dt
    apply_bcs!(solver, Q)
    return Q
end

"""
    compute_dt(solver, Q)

CFL-limited timestep from the acoustic (|u_d| + c) / (h_d · scalefactor_d)
rate and a diffusive rate built from the current (possibly artificial)
transport coefficients, reduced over all ranks. Calls `primitives!` so the
estimate is EOS-aware; artificial coefficients lag by one step.
"""
function compute_dt(solver::Solver, Q)
    decomp = solver.decomp
    exchange_state!(Q, decomp)   # keep halos consistent for primitives!
    primitives!(solver, Q)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    tr = solver.transport
    rate = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = solver.rho[I]
        ri = 1 / ρ
        c = solver.c[I]
        cp = solver.cp_mix[I]
        uv = (solver.u[I], solver.v[I], solver.w[I])
        acc = 0.0
        dsum = 0.0
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
        for sp in 1:solver.n_species
            Dmax = max(Dmax, tr.mu0 / (tr.Sc * ρ) + solver.D_art[sp][I])
        end
        ν = (tr.mu0 + solver.mu_art[I] + solver.beta_art[I]) * ri +
            (tr.mu0 * cp / tr.Pr + solver.kappa_art[I]) * ri / cp + Dmax
        acc += 2 * ν * dsum
        rate = max(rate, acc)
    end
    rate = MPI.Allreduce(rate, max, decomp.comm)
    return solver.cfl / rate
end

"""
    curvature_rate(solver, metric, I, uv)

Rate contribution from geometric momentum sources on angular dimensions that
are collapsed (and therefore contribute no advective CFL term). Zero in
Cartesian coordinates and whenever the corresponding dimension is resolved.
"""
curvature_rate(solver, ::CartesianMetric, I, uv) = 0.0

function curvature_rate(solver, ::CylindricalMetric, I, uv)
    solver.decomp.active[2] && return 0.0          # covered by the θ advective term
    return abs(uv[2]) * solver.inv_r[I]          # ρu_θ²/r driving u_r
end

function curvature_rate(solver, ::SphericalMetric, I, uv)
    a = 0.0
    solver.decomp.active[2] || (a += abs(uv[2]) * solver.inv_r[I])
    solver.decomp.active[3] ||
        (a += abs(uv[3]) * (solver.inv_r[I] + abs(solver.cot_over_r[I])))
    return a
end

"""
    dt_report(solver, Q)

Diagnostic companion to `compute_dt`: returns a NamedTuple naming the global
timestep limiter — `(dt, rank, index, coords, dim, kind)` where `kind` is
`:acoustic`, `:diffusive`, or `:curvature`. Cheap enough to call every few
hundred steps; the intended use is confirming whether a run is limited by
the physics you care about or by the azimuthal spacing at a coordinate
singularity (see the CFL notes in the README).
"""
function dt_report(solver::Solver, Q)
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
        for sp in 1:solver.n_species
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
    grate = MPI.Allreduce(best.rate, max, decomp.comm)
    mine = best.rate >= grate ? MPI.Comm_rank(decomp.comm) : typemax(Int)
    owner = MPI.Allreduce(mine, min, decomp.comm)
    return (dt=solver.cfl / grate, rank=owner, index=(best.i, best.j, best.k),
            coords=(xcoord(solver, 1, best.i), xcoord(solver, 2, best.j),
                    xcoord(solver, 3, best.k)),
            dim=best.dim, kind=best.kind)
end

"""
    run!(solver, Q; tfinal, nmax=typemax(Int), callback=nothing)

Advance to `tfinal` (or `nmax` steps), filtering the conserved variables every
`s.filter_interval` steps and invoking `callback(solver, Q)` after each step.
"""
function run!(solver::Solver, Q; tfinal, nmax::Int=typemax(Int), callback=nothing)
    dQ = zero(Q)
    du = zero(Q)
    while solver.t < tfinal && solver.step < nmax
        dt = min(compute_dt(solver, Q), tfinal - solver.t)
        step!(solver, Q, dQ, du, dt)
        solver.t += dt
        solver.step += 1
        if solver.filter_interval > 0 && solver.step % solver.filter_interval == 0
            filter_state!(solver, Q)
        end
        callback !== nothing && callback(solver, Q)
    end
    return Q
end

"""
    filter_state!(solver, Q)

Apply the compact filter to every conserved component along every active
dimension, with batched per-dimension halo exchange and axis parity routing
(ρu_r and ρu_θ are odd across the axis; everything else is even).
"""
function filter_state!(solver::Solver, Q)
    decomp = solver.decomp
    comps = [view(Q, :, :, :, c) for c in 1:solver.n_cons]
    for d in 1:3
        decomp.active[d] || continue
        exchange_dim_batch!(comps, decomp, d)
        for c in 1:solver.n_cons
            filt_along!(solver.tmp_a, comps[c], solver, d, cons_parity(solver, d, c))
            copy_interior!(comps[c], solver.tmp_a, decomp)
        end
    end
    return Q
end
