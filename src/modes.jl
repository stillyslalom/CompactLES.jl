# Azimuthal mode truncation near a cylindrical axis.
#
# On a resolved-θ grid the azimuthal spacing r·Δθ at the first radial nodes
# is tighter than Δr by about N_θ/π, and `max_rate` charges the acoustic
# rate at that spacing and the diffusive rate at its square. A field that is
# smooth through the axis carries azimuthal mode m only as r^m, so the rings
# near the axis resolve few of the N_θ/2 modes the grid carries. The
# truncation projects each such ring (fixed r and z) onto its modes
# m ≤ mode_limit once per step, and `max_rate` and `dt_report` charge the
# θ direction at the spacing of the highest retained mode. Both read the
# one table built here, so the charged and the removed sets cannot differ.
#
# The projection is direct, against cos/sin tables of the N_θ ring angles,
# O(N_θ · mode_limit) per ring and component: the limits are small wherever
# the truncation is active, so no FFT is warranted, and the projection is
# exact, not an approximation of one. The m = 0 coefficient is kept, so ring
# sums of every conserved component (hence mass and energy, the θ quadrature
# weights being uniform and the metric factors independent of θ) are
# preserved, and mode_limit ≥ 1 keeps the m = ±1 content of the physical
# momenta, which carries the Cartesian momentum and a uniform freestream.
# Fourier modes have parity (−1)^m under θ → θ + π, so the projection
# commutes with the antipodal fold (folds.jl), which pairs radial lines and
# is never touched by a ring at r > 0.

"""
    ModeTruncation{T}

The azimuthal truncation table of a cylindrical solver with resolved θ,
built at setup from `Numerics(polar_truncation = κ)`. `rings` are the padded
radial indices of this rank's active rings, those whose limit

    mode_limit(r) = max(1, floor(π r / (κ Δr)))

falls below N_θ/2, and `mode_limit` and `theta_cap = mode_limit / (π r)`
hold the limit and the capped inverse θ spacing of each. `cosines` and
`sines` tabulate cos(2πn/N_θ) and sin(2πn/N_θ), `coef` is the host scratch
of one ring's coefficients. With `kappa == 0` every table is empty and the
feature is off; see `truncate_modes!`.
"""
struct ModeTruncation{T}
    kappa::T                        # 0 = off
    rings::UnitRange{Int}           # padded radial indices of the active rings
    mode_limit::Vector{Int}         # per active ring: the highest retained mode
    theta_cap::Vector{T}            # per active ring: mode_limit / (π r)
    cosines::Vector{T}              # cos(2π n / N_θ), n = 0 … N_θ − 1
    sines::Vector{T}
    coef::Vector{T}                 # one ring's a₀, (a_m, b_m) …; host scratch
end

ModeTruncation{T}() where {T} =
    ModeTruncation{T}(zero(T), 1:0, Int[], T[], T[], T[], T[])

# The table over this rank's radial block. `r0` is the physical radius of
# global radial node 1 and `Δr` the uniform radial spacing; the limit is
# evaluated in Float64 once here, and every consumer reads the integers.
function mode_truncation(::Type{T}, kappa, decomp::Decomp, r0, Δr,
                         n_theta::Int) where {T}
    kappa > 0 || return ModeTruncation{T}()
    κ = Float64(kappa)
    o1 = decomp.n_halo_d[1]
    limits = Int[]
    caps = T[]
    for i in 1:decomp.n_local[1]
        r = Float64(r0) + (decomp.offset[1] + i - 1) * Float64(Δr)
        m = max(1, floor(Int, π * r / (κ * Float64(Δr))))
        2m < n_theta || break          # the limit grows with r: the rest keep all
        push!(limits, m)
        push!(caps, T(m / (π * r)))
    end
    rings = (o1 + 1):(o1 + length(limits))
    n = 0:(n_theta - 1)
    cosines = T[cospi(2k / n_theta) for k in n]
    sines = T[sinpi(2k / n_theta) for k in n]
    mmax = isempty(limits) ? 0 : maximum(limits)
    return ModeTruncation{T}(T(kappa), rings, limits, caps, cosines, sines,
                             zeros(T, 2mmax + 1))
end

# Whether the feature is on. Replicated: κ is a setup keyword.
@inline _truncating(tr::ModeTruncation) = tr.kappa > 0

# The θ inverse spacing `max_rate` and `dt_report` charge at padded radial
# index `ir`: the grid's own `idx`, capped at an active ring to the spacing
# of the ring's highest retained mode.
@inline function _theta_spacing(tr::ModeTruncation, ir::Int, idx)
    ir in tr.rings || return idx
    return @inbounds min(idx, tr.theta_cap[ir - first(tr.rings) + 1])
end

"""
    truncate_modes!(solver, Q) -> Q

Project every conserved component of `Q` on each active ring of the
solver's azimuthal truncation table onto the ring's Fourier modes
`m ≤ mode_limit`, over the interior only. A no-op unless the solver was
built with `polar_truncation > 0`. [`run!`](@ref) calls it once per step,
after the state filter and before the positivity failsafe; halos are left
stale, as they are after the failsafe, and the next `max_rate` exchanges
them. Rank-local and free of collectives, since setup keeps θ on one rank.

The projection is idempotent to round-off, preserves the ring sum of every
component, and keeps modes 0 and 1, hence a uniform freestream and the
Cartesian momentum of each ring.
"""
function truncate_modes!(solver, Q::ConservedState)
    tr = solver.truncation
    _truncating(tr) || return Q
    _truncate_rings!(parent(Q), tr, solver.decomp, solver.equations.n_cons)
    return Q
end

truncate_modes!(solver, states::Vector{<:ConservedState}) =
    (_truncating(solver.truncation) &&
     error("polar_truncation takes a single-patch solver"); states)

function _truncate_rings!(q::AbstractArray{T,4}, tr::ModeTruncation{T},
                          decomp::Decomp, n_cons::Int) where {T}
    o2, o3 = decomp.n_halo_d[2], decomp.n_halo_d[3]
    nz = decomp.n_local[3]
    N = length(tr.cosines)
    cs = tr.cosines
    sn = tr.sines
    a = tr.coef
    invN = one(T) / N
    @inbounds for comp in 1:n_cons, k in (o3 + 1):(o3 + nz),
                  (slot, ir) in enumerate(tr.rings)
        M = tr.mode_limit[slot]
        s = zero(T)
        for j in 1:N
            s += q[ir, j + o2, k, comp]
        end
        a[1] = s * invN
        for m in 1:M
            sc = zero(T)
            ss = zero(T)
            n = 0                              # m (j − 1) mod N
            for j in 1:N
                x = q[ir, j + o2, k, comp]
                sc += x * cs[n + 1]
                ss += x * sn[n + 1]
                n += m
                n >= N && (n -= N)
            end
            a[2m] = 2 * sc * invN
            a[2m + 1] = 2 * ss * invN
        end
        for j in 1:N
            x = a[1]
            n = 0                              # m (j − 1) mod N
            for m in 1:M
                n += j - 1
                n >= N && (n -= N)
                x += a[2m] * cs[n + 1] + a[2m + 1] * sn[n + 1]
            end
            q[ir, j + o2, k, comp] = x
        end
    end
    return q
end
