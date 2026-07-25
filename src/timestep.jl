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
    step!(s, Q, dQ, du, dt)

Advance the conserved state by one RK45 step of size `dt`. `dQ` and `du` are
caller-provided work arrays of the same shape as `Q`.
"""
function step!(s::Solver, Q, dQ, du, dt)
    dec = s.dec
    o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    for stage in 1:5
        s.tstage = s.t + RKC[stage] * dt
        apply_bcs!(s, Q)
        compute_rhs!(s, Q, dQ)
        A = RKA[stage]
        B = RKB[stage]
        for c in 1:s.ncons
            Threads.@threads for k in 1:nz
                @inbounds for j in 1:ny, i in 1:nx
                    v = A * du[i+o1, j+o2, k+o3, c] + dt * dQ[i+o1, j+o2, k+o3, c]
                    du[i+o1, j+o2, k+o3, c] = v
                    Q[i+o1, j+o2, k+o3, c] += B * v
                end
            end
        end
    end
    s.tstage = s.t + dt
    apply_bcs!(s, Q)
    return Q
end

"""
    compute_dt(s, Q)

CFL-limited timestep from the acoustic (|u_d| + c) / (h_d · scalefactor_d)
rate and a diffusive rate built from the current (possibly artificial)
transport coefficients, reduced over all ranks. Calls `primitives!` so the
estimate is EOS-aware; artificial coefficients lag by one step.
"""
function compute_dt(s::Solver, Q)
    dec = s.dec
    exchange_state!(Q, dec)   # keep halos consistent for primitives!
    primitives!(s, Q)
    o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    tr = s.transport
    rate = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = s.rho[I]
        ri = 1 / ρ
        c = s.c[I]
        uv = (s.u[I], s.v[I], s.w[I])
        acc = 0.0
        dsum = 0.0
        for d in 1:3
            dec.active[d] || continue        # no resolved variation
            idx = s.invh[d][I] / s.h[d]      # inverse physical spacing
            acc += (abs(uv[d]) + c) * idx
            dsum += idx * idx
        end
        # Curvature-source stiffness. When an angular dimension is RESOLVED,
        # its source rate (|u_ang|/r) is smaller than its advective rate
        # (|u_ang|/(r Δang)) and is already covered above. When it is
        # COLLAPSED — axisymmetric-with-swirl being the important case — the
        # loop skips it entirely, yet ρu_θ²/r still drives u_r as a stiff
        # source at small r. That term is added here.
        acc += curvature_rate(s, s.metric, I, uv)
        Dmax = tr.mu0 / (tr.Sc * ρ)
        for sp in 1:s.ns
            Dmax = max(Dmax, tr.mu0 / (tr.Sc * ρ) + s.Dart[sp][I])
        end
        ν = (tr.mu0 + s.mua[I] + s.betaa[I]) * ri +
            (tr.mu0 * s.cpm[I] / tr.Pr + s.kappaa[I]) * ri / s.cpm[I] + Dmax
        acc += 2 * ν * dsum
        rate = max(rate, acc)
    end
    rate = MPI.Allreduce(rate, max, dec.comm)
    return s.cfl / rate
end

"""
    curvature_rate(s, metric, I, uv)

Rate contribution from geometric momentum sources on angular dimensions that
are collapsed (and therefore contribute no advective CFL term). Zero in
Cartesian coordinates and whenever the corresponding dimension is resolved.
"""
curvature_rate(s, ::CartesianMetric, I, uv) = 0.0

function curvature_rate(s, ::CylindricalMetric, I, uv)
    s.dec.active[2] && return 0.0          # covered by the θ advective term
    return abs(uv[2]) * s.rinv[I]          # ρu_θ²/r driving u_r
end

function curvature_rate(s, ::SphericalMetric, I, uv)
    a = 0.0
    s.dec.active[2] || (a += abs(uv[2]) * s.rinv[I])
    s.dec.active[3] || (a += abs(uv[3]) * (s.rinv[I] + abs(s.cotr[I])))
    return a
end

"""
    dt_report(s, Q)

Diagnostic companion to `compute_dt`: returns a NamedTuple naming the global
timestep limiter — `(dt, rank, index, coords, dim, kind)` where `kind` is
`:acoustic`, `:diffusive`, or `:curvature`. Cheap enough to call every few
hundred steps; the intended use is confirming whether a run is limited by
the physics you care about or by the azimuthal spacing at a coordinate
singularity (see the CFL notes in the README).
"""
function dt_report(s::Solver, Q)
    dec = s.dec
    exchange_state!(Q, dec)
    primitives!(s, Q)
    o1, o2, o3 = dec.Hd
    tr = s.transport
    best = (rate=-Inf, i=0, j=0, k=0, dim=0, kind=:none)
    @inbounds for k in 1:dec.nloc[3], j in 1:dec.nloc[2], i in 1:dec.nloc[1]
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = s.rho[I]; ri = 1 / ρ; c = s.c[I]
        uv = (s.u[I], s.v[I], s.w[I])
        acc = 0.0; dsum = 0.0; wdim = 0; wrate = -Inf
        for d in 1:3
            dec.active[d] || continue
            idx = s.invh[d][I] / s.h[d]
            rd = (abs(uv[d]) + c) * idx
            acc += rd; dsum += idx * idx
            rd > wrate && (wrate = rd; wdim = d)
        end
        crate = curvature_rate(s, s.metric, I, uv)
        Dmax = tr.mu0 / (tr.Sc * ρ)
        for sp in 1:s.ns
            Dmax = max(Dmax, tr.mu0 / (tr.Sc * ρ) + s.Dart[sp][I])
        end
        ν = (tr.mu0 + s.mua[I] + s.betaa[I]) * ri +
            (tr.mu0 * s.cpm[I] / tr.Pr + s.kappaa[I]) * ri / s.cpm[I] + Dmax
        drate = 2 * ν * dsum
        total = acc + crate + drate
        if total > best.rate
            kind = drate > max(wrate, crate) ? :diffusive :
                   crate > wrate ? :curvature : :acoustic
            best = (rate=total, i=i, j=j, k=k, dim=wdim, kind=kind)
        end
    end
    grate = MPI.Allreduce(best.rate, max, dec.comm)
    mine = best.rate >= grate ? MPI.Comm_rank(dec.comm) : typemax(Int)
    owner = MPI.Allreduce(mine, min, dec.comm)
    return (dt=s.cfl / grate, rank=owner, index=(best.i, best.j, best.k),
            coords=(xcoord(s, 1, best.i), xcoord(s, 2, best.j),
                    xcoord(s, 3, best.k)),
            dim=best.dim, kind=best.kind)
end

"""
    run!(s, Q; tfinal, nmax=typemax(Int), callback=nothing)

Advance to `tfinal` (or `nmax` steps), filtering the conserved variables every
`s.filter_interval` steps and invoking `callback(s, Q)` after each step.
"""
function run!(s::Solver, Q; tfinal, nmax::Int=typemax(Int), callback=nothing)
    dQ = zero(Q)
    du = zero(Q)
    while s.t < tfinal && s.step < nmax
        dt = min(compute_dt(s, Q), tfinal - s.t)
        step!(s, Q, dQ, du, dt)
        s.t += dt
        s.step += 1
        if s.filter_interval > 0 && s.step % s.filter_interval == 0
            filter_state!(s, Q)
        end
        callback !== nothing && callback(s, Q)
    end
    return Q
end

"""
    filter_state!(s, Q)

Apply the compact filter to every conserved component along every active
dimension, with batched per-dimension halo exchange and axis parity routing
(ρu_r and ρu_θ are odd across the axis; everything else is even).
"""
function filter_state!(s::Solver, Q)
    dec = s.dec
    comps = [view(Q, :, :, :, c) for c in 1:s.ncons]
    for d in 1:3
        dec.active[d] || continue
        exchange_dim_batch!(comps, dec, d)
        for c in 1:s.ncons
            filt_along!(s.tmpA, comps[c], s, d, cons_parity(s, d, c))
            copy_interior!(comps[c], s.tmpA, dec)
        end
    end
    return Q
end
