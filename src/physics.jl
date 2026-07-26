# Thermodynamics: the EOS contract, and three models implementing it.
#
# The single temperature is named T_ion throughout, leaving room for T_ele /
# T_rad when this grows a 2T or 3T model.
#
# The conserved layout is (ρY₁..ρY_Ns, ρu, ρv, ρw, E); mixture density is the
# sum of partial densities. Hot loops reach the EOS through a function barrier
# (`primitives!(solver, Q)` dispatches on `typeof(solver.eos)`), so an abstractly
# typed `eos` field costs one dynamic dispatch per array pass, not per point.
#
# --- The contract ----------------------------------------------------------
#
# A new EOS supplies these and nothing else. The first four are the ones that
# were always here; the last three existed as ideal-gas algebra inlined at their
# call sites, which is what made the rest of the solver quietly ideal-gas even
# though this file was not.
#
#   nspecies(eos)                        number of species Ns
#   _primitives!(solver, eos, Q)         ρ, u, p, T_ion, c, cp_mix, Y over the
#                                        padded arrays — the bulk conversion
#   species_enthalpy(eos, k, T_ion)      partial specific enthalpy h_k(T_ion)
#   conserved_from_prim(eqns, eos, pr)   primitive → conserved (problem.jl)
#
#   eos_phi(eos, ρ, p, T_ion, cp_mix)    φ = ∂(ρe)/∂p |_{ρ,Y}, the coefficient
#                                        that maps a pressure wave amplitude
#                                        onto total energy. NSCBC needs it for
#                                        both faces.
#   eos_dphi_dY(eos, k, ρ, p, T, cp)     ∂φ/∂Y_k, for the species terms of the
#                                        characteristic inflow.
#   art_conductivity_scale(eos, ρ, c,    κ* per unit sensor in compute_artificial!
#                          T_ion, cp)
#   wall_internal_energy(eos, Q, I,      ρe at a wall held at Twall (boundary.jl)
#                        n_species, Twall)
#
# Every one of them is called from behind a function barrier, once per array
# pass or once per boundary plane — never per point through an abstract field.
#
# --- Why φ and the κ* scale are in the contract ----------------------------
#
# φ = ∂(ρe)/∂p is what turns the LODI wave analysis in nscbc.jl from ideal-gas
# algebra into an EOS query. For an ideal mixture φ = cv_m/R_m; for a stiffened
# gas it is 1/(γ−1) with no composition dependence at all; for a table it is a
# derivative of the table. The wave amplitudes themselves are EOS-independent —
# it is only the projection onto the conserved energy that is not.
#
# The κ* scale is the weaker of the two abstractions and is worth being explicit
# about. Cook's artificial conductivity is κ* = C_κ (ρc/T_ion)·sensor, which is
# an ideal-gas construction twice over: it assumes e ∝ T, and it is singular as
# T_ion → 0. The singularity is real and reachable — a cold ambient at p ≲ 1e-3
# drives the diffusive timestep to collapse (reference/CALIBRATION.md has the
# measurement). Making it a dispatch point does not fix that; it makes the
# assumption visible and lets a condensed-matter or tabular EOS supply a scale
# that is finite at its own cold limit, which is the prerequisite the roadmap
# names. Fixing the singularity for ideal gases is a separate numerics decision.

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
species_enthalpy(eos::IdealMixture, k::Int, T_ion) = eos.cpk[k] * T_ion

# φ = cv_m/R_m, recovered from the stored primitives rather than from Y: the
# mixture gas constant is R_m = p/(ρT_ion) by definition of the state, and
# cv_m = cp_m − R_m. That keeps the signature identical for every EOS.
@inline function eos_phi(::IdealMixture, ρ, p, T_ion, cp_mix)
    Rm = p / (ρ * T_ion)
    return cp_mix / Rm - 1
end

"∂(cv_m/R_m)/∂Y_k for ideal mixtures."
@inline function eos_dphi_dY(eos::IdealMixture, k::Int, ρ, p, T_ion, cp_mix)
    Rm = p / (ρ * T_ion)
    cvm = cp_mix - Rm
    return (eos.cvk[k] * Rm - eos.Rk[k] * cvm) / (Rm * Rm)
end

"Cook's ρc/T_ion. See the note on the singularity at the top of this file."
@inline art_conductivity_scale(::IdealMixture, ρ, c, T_ion, cp_mix) =
    ρ * c / max(T_ion, eps(typeof(T_ion)))

# ---------------------------------------------------------------------------
# Stiffened gas: the minimal condensed-matter EOS.
#
#   p = (γ − 1) ρ e − γ p∞
#
# with p∞ the cohesive pressure that makes a liquid stiff. Setting p∞ = 0
# recovers a perfect gas exactly, which is how the tests pin it down. The
# algebra is the perfect-gas algebra in (p + p∞):
#
#   p + p∞ = ρ R T_ion,  R = (γ − 1) c_v,   c² = γ (p + p∞) / ρ
#   e = c_v T_ion + p∞/ρ,  so  ρe = ρ c_v T_ion + p∞
#
# and the two derived quantities the contract asks for are unusually clean:
# φ = ∂(ρe)/∂p = 1/(γ − 1) exactly, independent of state, and there is no
# composition dependence because this is single-component.
#
# Water at ambient conditions is roughly γ = 4.4, p∞ = 6.0e8 Pa; aluminium and
# other metals are usually done with Mie–Grüneisen instead, which is the same
# shape of model with a density-dependent reference curve and is the natural
# next member of this family.

"""
    StiffenedGas(; gamma, p_inf, cv, name)

Single-component stiffened-gas equation of state, `p = (γ−1)ρe − γp∞`. With
`p_inf = 0` this is a perfect gas and reproduces [`single_species`](@ref)
exactly.
"""
Base.@kwdef struct StiffenedGas{T} <: EOS
    gamma::T = 4.4
    p_inf::T = 6.0e8
    cv::T    = 1816.0
    name::String = "liquid"
end

nspecies(::StiffenedGas) = 1
gas_constant(eos::StiffenedGas) = (eos.gamma - 1) * eos.cv
species_names(eos::StiffenedGas) = [eos.name]
species_enthalpy(eos::StiffenedGas, ::Int, T_ion) = eos.gamma * eos.cv * T_ion

# Exact and state-independent: ρe = (p + γp∞)/(γ−1).
@inline eos_phi(eos::StiffenedGas, ρ, p, T_ion, cp_mix) = 1 / (eos.gamma - 1)
@inline eos_dphi_dY(::StiffenedGas, ::Int, ρ, p, T_ion, cp_mix) = 0.0
@inline art_conductivity_scale(::StiffenedGas, ρ, c, T_ion, cp_mix) =
    ρ * c / max(T_ion, eps(typeof(T_ion)))

function _primitives!(solver, eos::StiffenedGas, Q)
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    γ = eos.gamma; p_inf = eos.p_inf; cv = eos.cv
    R = gas_constant(eos)
    ρa, ua, va, wa = solver.rho, solver.u, solver.v, solver.w
    pa, T_iona, ca, cpa = solver.p, solver.T_ion, solver.c, solver.cp_mix
    nxf, nyf, nzf = size(ρa)
    @threaded nxf*nyf*nzf for k in 1:nzf
        @inbounds for j in 1:nyf, i in 1:nxf
            ρ = Q[i, j, k, 1]
            if ρ > 0
                ri = 1 / ρ
                u = Q[i, j, k, m1] * ri
                v = Q[i, j, k, m2] * ri
                w = Q[i, j, k, m3] * ri
                e = Q[i, j, k, i_energy] * ri - 0.5 * (u*u + v*v + w*w)
                p = (γ - 1) * ρ * e - γ * p_inf
                ρa[i, j, k] = ρ
                ua[i, j, k] = u; va[i, j, k] = v; wa[i, j, k] = w
                pa[i, j, k] = p
                T_iona[i, j, k] = max((p + p_inf) / (ρ * R), 1e-300)
                ca[i, j, k] = sqrt(max(γ * (p + p_inf) * ri, 0.0))
                cpa[i, j, k] = γ * cv
                solver.Y[1][i, j, k] = 1.0
            else
                ρa[i, j, k] = 1
                ua[i, j, k] = 0; va[i, j, k] = 0; wa[i, j, k] = 0
                pa[i, j, k] = 1; T_iona[i, j, k] = 1; ca[i, j, k] = 1
                cpa[i, j, k] = γ * cv
                solver.Y[1][i, j, k] = 1.0
            end
        end
    end
    return solver
end

# ---------------------------------------------------------------------------
# NASA-9 polynomial mixture: temperature-dependent specific heats.
#
# The standard form (McBride, Zehe & Gordon 2002), per species, on one
# temperature interval:
#
#   cp/R = a₁T⁻² + a₂T⁻¹ + a₃ + a₄T + a₅T² + a₆T³ + a₇T⁴
#   h/(RT) = −a₁T⁻² + a₂ln(T)/T + a₃ + a₄T/2 + a₅T²/3 + a₆T³/4 + a₇T⁴/5 + b₁/T
#
# with the h expression the exact integral of the cp one, which is the property
# the consistency test checks numerically rather than trusting the transcription.
#
# NO COEFFICIENT DATABASE IS SHIPPED. Real NASA-9 coefficients are tabulated per
# species over two or three temperature intervals and belong in a data file
# under the user's control, not transcribed into a solver from memory — a wrong
# digit in a₃ is a wrong γ that no test here would catch. `Nasa9Species` takes
# the coefficients directly; `nasa9_constant_cp` builds the degenerate
# constant-cp case, which is what the tests use to pin the machinery against
# `IdealMixture` exactly.
#
# The cost of a caloric EOS is that T is no longer explicit in e: `_primitives!`
# runs a Newton iteration per point, converging on de/dT = cv. That is inherent
# to the model, not to this implementation.

"""
    Nasa9Species(name, R, a, b1; Tmin, Tmax)

One species of a [`Nasa9Mixture`](@ref). `a` is the seven-term NASA-9 heat
capacity vector and `b1` the enthalpy integration constant, both in the
convention documented above; `R` is the specific gas constant.
"""
Base.@kwdef struct Nasa9Species{T}
    name::String
    R::T
    a::NTuple{7,T}
    b1::T = zero(T)
    Tmin::T = 200.0
    Tmax::T = 6000.0
end

"""
    nasa9_constant_cp(name, R, cp)

Degenerate `Nasa9Species` with a temperature-independent cp — the reduction
that must reproduce an ideal-gas species exactly.
"""
nasa9_constant_cp(name::String, R::Real, cp::Real) =
    Nasa9Species{Float64}(name=name, R=Float64(R),
                          a=(0.0, 0.0, Float64(cp / R), 0.0, 0.0, 0.0, 0.0),
                          b1=0.0)

"""
    Nasa9Mixture(species)

Multicomponent mixture with NASA-9 polynomial heat capacities. Ideal in the
thermal sense (p = ρR_mT_ion) but not calorically: cp, cv and γ all vary with
temperature, which is what a real diatomic or triatomic gas does above a few
hundred kelvin and what a constant-γ mixture gets wrong at shock temperatures.
"""
struct Nasa9Mixture{T} <: EOS
    sp::Vector{Nasa9Species{T}}
    Rk::Vector{T}
    T_guess::T                 # Newton start when no better estimate exists
end

Nasa9Mixture(sp::Vector{Nasa9Species{T}}; T_guess=300.0) where {T} =
    Nasa9Mixture{T}(sp, [x.R for x in sp], T(T_guess))

nspecies(eos::Nasa9Mixture) = length(eos.sp)
species_names(eos::Nasa9Mixture) = [x.name for x in eos.sp]

"cp_k(T_ion) from the NASA-9 polynomial."
@inline function species_cp(eos::Nasa9Mixture, k::Int, T_ion)
    s = eos.sp[k]; a = s.a; t = T_ion
    return s.R * (a[1] / (t * t) + a[2] / t + a[3] +
                  a[4] * t + a[5] * t^2 + a[6] * t^3 + a[7] * t^4)
end

"h_k(T_ion), the exact integral of `species_cp`."
@inline function species_enthalpy(eos::Nasa9Mixture, k::Int, T_ion)
    s = eos.sp[k]; a = s.a; t = T_ion
    return s.R * t * (-a[1] / (t * t) + a[2] * log(t) / t + a[3] +
                      a[4] * t / 2 + a[5] * t^2 / 3 + a[6] * t^3 / 4 +
                      a[7] * t^4 / 5 + s.b1 / t)
end

"e_k(T_ion) = h_k − R_k T_ion."
@inline species_energy(eos::Nasa9Mixture, k::Int, T_ion) =
    species_enthalpy(eos, k, T_ion) - eos.Rk[k] * T_ion

"""
    mixture_temperature(eos, e, Yat) -> T_ion

Invert Σ_k Y_k e_k(T) = e for the temperature. `Yat` is a function of the
species index, so the caller can supply mass fractions from a tuple, a vector,
or straight out of the conserved array without materializing anything.

Newton on f(T) = Σ Y_k e_k(T) − e, whose derivative is the mixture cv and is
positive for any physical coefficient set, so the iteration is monotone. The
step is clamped so it cannot cross into T ≤ 0 from a garbage state in a stale
halo, and the iteration count is capped.

The starting point is a pure function of the state, deliberately: seeding from
the previous value at the same point is a percent away and would save an
iteration, but it makes the answer depend on call history, and the bit-for-bit
agreement between a serial and a decomposed run is the correctness oracle the
MPI suite is built on. That is not worth one Newton step.
"""
@inline function mixture_temperature(eos::Nasa9Mixture, e, Yat::F) where {F}
    n = length(eos.sp)
    # Seed from the constant-cv estimate at the reference temperature.
    cv0 = 0.0
    for k in 1:n
        cv0 += Yat(k) * (species_cp(eos, k, eos.T_guess) - eos.Rk[k])
    end
    T_ion = cv0 > 0 ? clamp(e / cv0, 1e-3, 1e9) : eos.T_guess
    for _ in 1:30
        f = -e; cvm = 0.0
        for k in 1:n
            Yk = Yat(k)
            f += Yk * species_energy(eos, k, T_ion)
            cvm += Yk * (species_cp(eos, k, T_ion) - eos.Rk[k])
        end
        cvm > 0 || break
        δ = f / cvm
        T_ion = max(T_ion - δ, 0.25 * T_ion)     # never overshoot into T ≤ 0
        abs(δ) <= 1e-14 * T_ion && break
    end
    return T_ion
end

@inline function eos_phi(::Nasa9Mixture, ρ, p, T_ion, cp_mix)
    Rm = p / (ρ * T_ion)
    return cp_mix / Rm - 1
end

# ∂φ/∂Y_k at fixed T_ion, with φ = cv_m/R_m: both numerator and denominator are
# mass-fraction averages, so the quotient rule gives the same shape as the ideal
# mixture with the temperature-dependent cv_k in place of the constant.
@inline function eos_dphi_dY(eos::Nasa9Mixture, k::Int, ρ, p, T_ion, cp_mix)
    Rm = p / (ρ * T_ion)
    cvm = cp_mix - Rm
    cvk = species_cp(eos, k, T_ion) - eos.Rk[k]
    return (cvk * Rm - eos.Rk[k] * cvm) / (Rm * Rm)
end

@inline art_conductivity_scale(::Nasa9Mixture, ρ, c, T_ion, cp_mix) =
    ρ * c / max(T_ion, eps(typeof(T_ion)))

function _primitives!(solver, eos::Nasa9Mixture, Q)
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    Rk = eos.Rk
    ρa, ua, va, wa = solver.rho, solver.u, solver.v, solver.w
    pa, T_iona, ca, cpa = solver.p, solver.T_ion, solver.c, solver.cp_mix
    nxf, nyf, nzf = size(ρa)
    @threaded nxf*nyf*nzf for k in 1:nzf
        @inbounds for j in 1:nyf, i in 1:nxf
            ρ = 0.0
            for sp in 1:n_species
                ρ += Q[i, j, k, sp]
            end
            if ρ > 0
                ri = 1 / ρ
                Rm = 0.0
                for sp in 1:n_species
                    solver.Y[sp][i, j, k] = Q[i, j, k, sp] * ri
                    Rm += Q[i, j, k, sp] * ri * Rk[sp]
                end
                u = Q[i, j, k, m1] * ri
                v = Q[i, j, k, m2] * ri
                w = Q[i, j, k, m3] * ri
                e = Q[i, j, k, i_energy] * ri - 0.5 * (u*u + v*v + w*w)
                # Mass fractions straight out of Q: the Newton solve takes an
                # accessor rather than a vector so this stays allocation-free.
                T_ion = mixture_temperature(eos, e, sp -> Q[i, j, k, sp] * ri)
                cpm = 0.0
                for sp in 1:n_species
                    cpm += solver.Y[sp][i, j, k] * species_cp(eos, sp, T_ion)
                end
                cvm = cpm - Rm
                ρa[i, j, k] = ρ
                ua[i, j, k] = u; va[i, j, k] = v; wa[i, j, k] = w
                pa[i, j, k] = ρ * Rm * T_ion
                T_iona[i, j, k] = T_ion
                ca[i, j, k] = sqrt(max((cpm / cvm) * Rm * T_ion, 0.0))
                cpa[i, j, k] = cpm
            else
                ρa[i, j, k] = 1
                ua[i, j, k] = 0; va[i, j, k] = 0; wa[i, j, k] = 0
                pa[i, j, k] = 1; T_iona[i, j, k] = 1; ca[i, j, k] = 1
                cpa[i, j, k] = 1
                for sp in 1:n_species
                    solver.Y[sp][i, j, k] = sp == 1 ? 1.0 : 0.0
                end
            end
        end
    end
    return solver
end

"Molecular transport coefficients (constant-property draft): μ₀ plus Prandtl
and Schmidt numbers setting molecular conductivity and species diffusivity."
Base.@kwdef struct Transport{T}
    mu0::T = 0.0
    Pr::T  = 0.7
    Sc::T  = 0.7
end

"""
    primitives!(solver, Q)

Recover ρ, u, v, w, p, T_ion, c, cp_mix, and the species mass fractions Y_k over
the full padded arrays (rank-boundary halos of Q must be current). Stale
physical-edge halos get benign placeholders; they are never read.
"""
primitives!(solver, Q) = _primitives!(solver, solver.eos, Q)

function _primitives!(solver, eos::IdealMixture, Q)
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    Rk, cvk = eos.Rk, eos.cvk
    ρa, ua, va, wa = solver.rho, solver.u, solver.v, solver.w
    pa, T_iona, ca, cpa = solver.p, solver.T_ion, solver.c, solver.cp_mix
    nxf, nyf, nzf = size(ρa)
    @threaded nxf*nyf*nzf for k in 1:nzf
        @inbounds for j in 1:nyf, i in 1:nxf
            ρ = 0.0
            for sp in 1:n_species
                ρ += Q[i, j, k, sp]
            end
            if ρ > 0
                ri = 1 / ρ
                Rm = 0.0; cvm = 0.0
                for sp in 1:n_species
                    Rm  += Q[i, j, k, sp] * Rk[sp]
                    cvm += Q[i, j, k, sp] * cvk[sp]
                    solver.Y[sp][i, j, k] = Q[i, j, k, sp] * ri
                end
                Rm *= ri; cvm *= ri
                u = Q[i, j, k, m1] * ri
                v = Q[i, j, k, m2] * ri
                w = Q[i, j, k, m3] * ri
                e = Q[i, j, k, i_energy] * ri - 0.5 * (u*u + v*v + w*w)
                T_ion = max(e / cvm, 1e-300)
                p = ρ * Rm * T_ion
                γm = 1 + Rm / cvm
                ρa[i, j, k] = ρ
                ua[i, j, k] = u; va[i, j, k] = v; wa[i, j, k] = w
                pa[i, j, k] = p
                T_iona[i, j, k] = T_ion
                ca[i, j, k] = sqrt(γm * Rm * T_ion)
                cpa[i, j, k] = cvm + Rm
            else
                ρa[i, j, k] = 1
                ua[i, j, k] = 0; va[i, j, k] = 0; wa[i, j, k] = 0
                pa[i, j, k] = 1; T_iona[i, j, k] = 1; ca[i, j, k] = 1
                cpa[i, j, k] = 1
                for sp in 1:n_species
                    solver.Y[sp][i, j, k] = sp == 1 ? 1.0 : 0.0
                end
            end
        end
    end
    return solver
end
