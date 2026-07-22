# Thermodynamics: EOS abstraction with a multicomponent ideal-gas mixture.
#
# The solver talks to the EOS through a small contract designed so that cubic
# (Peng–Robinson etc.), polynomial-cp, or tabular models can be dropped in
# later without touching the flow solver:
#
#   nspecies(eos)                      → number of species Ns
#   eos_state(eos, ρ, e, Yweights)     → (p, T, c, cp_mix); Yweights supplies
#                                        Y_k on demand via a closure/functor
#   species_enthalpy(eos, k, T)        → partial specific enthalpy h_k(T)
#
# The conserved layout is (ρY₁..ρY_Ns, ρu, ρv, ρw, E); mixture density is the
# sum of partial densities. Hot loops reach the EOS through a function barrier
# (`primitives!(s, Q)` dispatches on `typeof(s.eos)`), so an abstractly typed
# `eos` field costs one dynamic dispatch per array pass, not per point.

abstract type EOS end

struct IdealSpecies{T}
    name::String
    R::T          # specific gas constant
    gamma::T
end

struct IdealMixture{T} <: EOS
    sp::Vector{IdealSpecies{T}}
    Rk::Vector{T}
    cvk::Vector{T}
    cpk::Vector{T}
end

function IdealMixture(sp::Vector{IdealSpecies{T}}) where {T}
    Rk  = [x.R for x in sp]
    cvk = [x.R / (x.gamma - 1) for x in sp]
    IdealMixture{T}(sp, Rk, cvk, Rk .+ cvk)
end

"Convenience constructor for a single calorically perfect gas."
single_species(; gamma::Real=1.4, R::Real=1.0, name::String="gas") =
    IdealMixture([IdealSpecies{Float64}(name, Float64(R), Float64(gamma))])

nspecies(eos::IdealMixture) = length(eos.sp)
species_enthalpy(eos::IdealMixture, k::Int, T) = eos.cpk[k] * T

"Molecular transport coefficients (constant-property draft): μ₀ plus Prandtl
and Schmidt numbers setting molecular conductivity and species diffusivity."
Base.@kwdef struct Transport{T}
    mu0::T = 0.0
    Pr::T  = 0.7
    Sc::T  = 0.7
end

"""
    primitives!(s, Q)

Recover ρ, u, v, w, p, T, c, cp_mix, and the species mass fractions Y_k over
the full padded arrays (rank-boundary halos of Q must be current). Stale
physical-edge halos get benign placeholders; they are never read.
"""
primitives!(s, Q) = _primitives!(s, s.eos, Q)

function _primitives!(s, eos::IdealMixture, Q)
    ns = s.ns
    m1, m2, m3 = s.mom
    ie = s.ie
    Rk, cvk = eos.Rk, eos.cvk
    ρa, ua, va, wa = s.rho, s.u, s.v, s.w
    pa, Ta, ca, cpa = s.p, s.Tt, s.c, s.cpm
    nxf, nyf, nzf = size(ρa)
    Threads.@threads for k in 1:nzf
        @inbounds for j in 1:nyf, i in 1:nxf
            ρ = 0.0
            for sp in 1:ns
                ρ += Q[i, j, k, sp]
            end
            if ρ > 0
                ri = 1 / ρ
                Rm = 0.0; cvm = 0.0
                for sp in 1:ns
                    Rm  += Q[i, j, k, sp] * Rk[sp]
                    cvm += Q[i, j, k, sp] * cvk[sp]
                    s.Ys[sp][i, j, k] = Q[i, j, k, sp] * ri
                end
                Rm *= ri; cvm *= ri
                u = Q[i, j, k, m1] * ri
                v = Q[i, j, k, m2] * ri
                w = Q[i, j, k, m3] * ri
                e = Q[i, j, k, ie] * ri - 0.5 * (u*u + v*v + w*w)
                T = max(e / cvm, 1e-300)
                p = ρ * Rm * T
                γm = 1 + Rm / cvm
                ρa[i, j, k] = ρ
                ua[i, j, k] = u; va[i, j, k] = v; wa[i, j, k] = w
                pa[i, j, k] = p
                Ta[i, j, k] = T
                ca[i, j, k] = sqrt(γm * Rm * T)
                cpa[i, j, k] = cvm + Rm
            else
                ρa[i, j, k] = 1
                ua[i, j, k] = 0; va[i, j, k] = 0; wa[i, j, k] = 0
                pa[i, j, k] = 1; Ta[i, j, k] = 1; ca[i, j, k] = 1
                cpa[i, j, k] = 1
                for sp in 1:ns
                    s.Ys[sp][i, j, k] = sp == 1 ? 1.0 : 0.0
                end
            end
        end
    end
    return s
end
