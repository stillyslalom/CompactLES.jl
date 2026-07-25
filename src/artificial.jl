# Cook-style artificial fluid properties (Cook, Phys. Fluids 2007), extended
# with the artificial species diffusivity for multicomponent mixing.
#
# Sensors use *undivided* explicit fourth differences δ⁴ (1, −4, 6, −4, 1) in
# the computational indices, so the formal grid-spacing powers reduce to
# per-dimension weights: h² for μ*, β* (from |∇⁴S| Δ⁶), h for κ* (|∇⁴e| Δ⁵)
# and for D* (|∇⁴Y| Δ⁵). The Gaussian test filter is approximated by one
# compact-filter pass. Species sensors are summed and a single common
# diffusivity is used, which with ΣY_k = 1 keeps Σ_k J_k = 0 identically
# (per-species D*_k with a correction velocity is a TODO). Indices are clamped
# at closed physical edges; halos cover rank boundaries. On curvilinear grids
# the sensors are computed in computational space — a grid-based rather than
# strictly physical-space regularization, standard practice and consistent
# with resolving power following the mesh.

Base.@kwdef struct ArtParams{T}
    enabled::Bool = true
    Cmu::T    = 0.002
    Cbeta::T  = 1.0
    Ckappa::T = 0.01
    CD::T     = 0.01
end

const D4 = (1.0, -4.0, 6.0, -4.0, 1.0)   # offsets −2:2

"""
    delta4_sum!(out, f, s, wpow; accumulate=false)

Interior accumulation of Σ_d h_d^wpow |δ⁴_d f| into `out` (added to existing
contents when `accumulate`). Requires current rank-boundary halos of `f`.
"""
function delta4_sum!(out, f, s, wpow::Int; accumulate::Bool=false)
    dec = s.dec
    o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    accumulate || fill!(out, 0)
    for d in 1:3
        dec.active[d] || continue
        wd = s.h[d]^wpow
        n_d = dec.nloc[d]
        lomin = at_lo_edge(dec, d) ? 1 : -1
        himax = at_hi_edge(dec, d) ? n_d : n_d + 2
        e = CartesianIndex(ntuple(k -> k == d ? 1 : 0, 3))
        @threaded nx*ny*nz for k in 1:nz
            @inbounds for j in 1:ny, i in 1:nx
                I = CartesianIndex(i + o1, j + o2, k + o3)
                il = (d == 1 ? i : d == 2 ? j : k)
                acc = 0.0
                for m in -2:2
                    ilm = clamp(il + m, lomin, himax)
                    acc += D4[m + 3] * f[I + (ilm - il) * e]
                end
                out[I] += wd * abs(acc)
            end
        end
    end
    return out
end

"One directional compact-filter pass per active dimension as a
Gaussian-filter proxy (even-parity axis fill along r when applicable)."
function smooth!(f, s)
    for d in 1:3
        s.dec.active[d] || continue
        # Only dimension d: the filter about to run is a 1-D stencil along d,
        # so the other two dimensions' halos are dead. This used to call
        # exchange_halos!, refreshing all three every time round the loop --
        # three times the traffic it needs in a 3-D run, on a routine that
        # runs once per sensor per species per RHS.
        exchange_dim!(f, s.dec, d)
        filt_along!(s.tmpA, f, s, d, 1)
        copy_interior!(f, s.tmpA, s.dec)
    end
    return f
end

"""
    compute_artificial!(s, Q)

Fill s.mua, s.betaa, s.kappaa, s.Dart from the current primitives and
(metric-corrected) velocity gradients. No-op if disabled.
"""
function compute_artificial!(s, Q)
    art = s.art
    if !art.enabled
        return s   # arrays stay zero from allocation
    end
    dec = s.dec
    o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    G = s.G

    # Strain-rate magnitude |S| = sqrt(S_ij S_ij) in the interior (physical
    # components — the metric corrections are already in G).
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ss = 0.0
            for b in 1:3, a in 1:3
                Sab = 0.5 * (G[a, b][I] + G[b, a][I])
                ss += Sab * Sab
            end
            s.Smag[I] = sqrt(ss)
        end
    end

    # μ*, β* sensor: Σ_d h_d² |δ⁴_d S|, smoothed.
    exchange_halos!(s.Smag, dec)
    delta4_sum!(s.sens, s.Smag, s, 2)
    smooth!(s.sens, s)
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ρsens = s.rho[I] * max(s.sens[I], 0.0)
            s.mua[I]   = art.Cmu   * ρsens
            s.betaa[I] = art.Cbeta * ρsens
        end
    end

    # κ* sensor: Σ_d h_d |δ⁴_d e|, smoothed; κ* = C_κ (ρ c / T) · sensor.
    # Internal energy directly from Q — EOS-agnostic.
    ie = s.ie
    m1, m2, m3 = s.mom
    @threaded length(s.tmpA) for k in 1:size(s.tmpA, 3)
        @inbounds for j in 1:size(s.tmpA, 2), i in 1:size(s.tmpA, 1)
            ρ = max(s.rho[i, j, k], 1e-300)
            ke = 0.5 * (Q[i,j,k,m1]^2 + Q[i,j,k,m2]^2 + Q[i,j,k,m3]^2) / ρ
            s.tmpA[i, j, k] = (Q[i, j, k, ie] - ke) / ρ
        end
    end
    exchange_halos!(s.tmpA, dec)
    delta4_sum!(s.sens, s.tmpA, s, 1)
    smooth!(s.sens, s)
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            s.kappaa[I] = art.Ckappa * s.rho[I] * s.c[I] /
                          max(s.Tt[I], eps()) * max(s.sens[I], 0.0)
        end
    end

    # Per-species D*_k sensors: Σ_d h_d |δ⁴_d Y_k|, each smoothed;
    # D*_k = C_D c · sensor_k. Costs ns filter sweeps per RHS; the flux
    # assembly's correction velocity keeps Σ_k J_k = 0 despite unequal D_k.
    # Only meaningful with more than one species.
    if s.ns > 1
        for sp in 1:s.ns
            delta4_sum!(s.sacc, s.Ys[sp], s, 1)
            smooth!(s.sacc, s)
            @threaded nx*ny*nz for k in 1:nz
                @inbounds for j in 1:ny, i in 1:nx
                    I = CartesianIndex(i + o1, j + o2, k + o3)
                    s.Dart[sp][I] = art.CD * s.c[I] * max(s.sacc[I], 0.0)
                end
            end
        end
    end
    return s
end
