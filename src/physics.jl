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
# A new EOS supplies these. `eos_phi`, `eos_dphi_dY` and
# `artificial_conductivity_scale`
# existed as ideal-gas algebra inlined at their call sites, which left the rest of
# the solver quietly ideal-gas even though this file was not; the others have been
# dispatch points throughout.
#
#   nspecies(eos)                        number of species Ns
#   recover_primitives!(solver, eos, Q)  ρ, u, v, w, p, T_ion, c, cp_mix, Y over
#                                        the padded arrays (the bulk conversion)
#   species_enthalpy(eos, k, T_ion)      partial specific enthalpy h_k(T_ion)
#   conserved_from_prim(eqns, eos, pr)   primitive → conserved (problem.jl)
#
#   eos_phi(eos, ρ, p, T_ion, cp_mix)    φ = ∂(ρe)/∂p |_{ρ,Y}, the coefficient
#                                        that maps a pressure wave amplitude
#                                        onto total energy. NSCBC needs it for
#                                        both faces.
#   eos_dphi_dY(eos, k, ρ, p, T, cp)     ∂φ/∂Y_k, for the species terms of the
#                                        characteristic inflow.
#   artificial_conductivity_scale(eos,   κ* per unit sensor in
#                          ρ, c,         compute_artificial!
#                          T_ion, cp)
#   wall_internal_energy(eos, Q, I,      ρe at a wall held at Twall (boundary.jl)
#                        n_species, Twall)
#   mole_fraction(eos, k, Y, I,          mole fraction X_k at index I from the
#                 n_species)             padded mass fractions Y (a FieldVector),
#                                        for the bulk species channel of
#                                        compute_artificial!
#   state_admissibility(eos, ρ, e, Yat,  whether a point is in the model's
#                       n_species)       thermodynamic domain, as STATE_ flags.
#                                        The state validation applies no
#                                        universal e > 0 test because the
#                                        internal-energy gauge is the EOS's.
#
# One further method is optional: `species_names(eos)` labels the partial-density
# components, and the fallback in equations.jl returns "species_1", "species_2"
# and so on for an EOS that does not define it.
#
# Several of these are evaluated per point: `species_enthalpy` in
# `_assemble_fluxes!`, `artificial_conductivity_scale` in `compute_artificial!`, and
# `wall_internal_energy` over an isothermal wall plane. Each such loop sits inside
# a routine specialized on the concrete solver or EOS type, so the dispatch on
# `eos` is paid once per array pass or once per boundary plane, not per point.
#
# --- Why φ and the κ* scale are in the contract ----------------------------
#
# φ = ∂(ρe)/∂p generalizes the LODI wave analysis in nscbc.jl from ideal-gas
# algebra into an EOS query. For an ideal mixture φ = cv_m/R_m; for a stiffened
# gas it is 1/(γ−1) with no composition dependence at all; for a table it is a
# derivative of the table. The wave amplitudes themselves are EOS-independent;
# only the projection onto the conserved energy is not.
#
# The κ* scale is the weaker of the two abstractions. Cook's artificial
# conductivity is κ* = C_κ (ρc/T_ion)·sensor, which is
# an ideal-gas construction twice over: it assumes e ∝ T, and it is singular as
# T_ion → 0. The singularity is reachable: a cold ambient at p ≲ 1e-3
# drives the diffusive timestep to collapse (reference/CALIBRATION.md has the
# measurement). Making it a dispatch point does not fix that; it makes the
# assumption visible and lets a condensed-matter or tabular EOS supply a scale
# that is finite at its own cold limit, which is the prerequisite the roadmap
# names. Fixing the singularity for ideal gases is a separate numerics decision.

"""
    EOS

Abstract equation-of-state interface mapping between conserved and primitive
thermodynamic variables. An EOS also fixes the species count and ordering used
by [`Prim`](@ref), the equation set, output, and species-indexed diagnostics.

Concrete implementations recover pressure, temperature, sound speed, heat
capacity, and mass fractions from a conserved state. They also provide
primitive-to-conserved conversion, species enthalpies, the thermodynamic
derivatives used by NSCBC, wall internal energy, and an artificial-conductivity
scale. Built-in choices are [`IdealMixture`](@ref), [`Nasa9Mixture`](@ref), and
[`StiffenedGas`](@ref).
"""
abstract type EOS end

"""
    recover_primitives!(solver, eos, Q) -> solver

Recover the padded primitive fields (`rho`, velocity, pressure, temperature,
sound speed, mixture heat capacity, and mass fractions) from the conserved
state `Q`. Custom [`EOS`](@ref) implementations must provide this bulk method.
It is called only after the state halos are current; [`refresh_primitives!`](@ref)
is the public collective wrapper for ordinary solver use.
"""
function recover_primitives! end

"""
    species_names(eos) -> Vector{String}

Labels for the partial-density components of an [`EOS`](@ref), in the same
order as its mass fractions. Defining this hook is optional; the equation-set
fallback names components `species_1`, `species_2`, and so on.
"""
function species_names end

"""
    species_enthalpy(eos, k, T_ion)

Specific enthalpy of species `k` at temperature `T_ion`. Custom [`EOS`](@ref)
implementations provide this hook for diffusive enthalpy fluxes.
"""
function species_enthalpy end

"""
    eos_phi(eos, rho, p, T_ion, cp_mix)

Return the thermodynamic NSCBC coefficient
`∂(rho*e)/∂p |_(rho,Y)` for an [`EOS`](@ref).
"""
function eos_phi end

"""
    eos_dphi_dY(eos, k, rho, p, T_ion, cp_mix)

Return the mass-fraction derivative of [`eos_phi`](@ref) for species `k`, at
fixed temperature. Custom [`EOS`](@ref) implementations provide this hook for
the composition terms of characteristic inflow.
"""
function eos_dphi_dY end

"""
    artificial_conductivity_scale(eos, rho, c, T_ion, cp_mix)

Return the EOS-dependent factor multiplying the artificial-conductivity sensor.
Custom [`EOS`](@ref) implementations provide this hook.
"""
function artificial_conductivity_scale end

"""
    wall_internal_energy(eos, Q, I, n_species, Twall)

Return the thermal conserved-energy density at padded index `I` when an
isothermal wall holds the state at `Twall`. Custom [`EOS`](@ref)
implementations provide this hook when they support [`NoSlipWallBC`](@ref)
with an isothermal condition.
"""
function wall_internal_energy end

"""
    IdealSpecies(name, R, gamma)
    IdealSpecies(name; R, gamma)
    IdealSpecies(name; T_ref=298.15, path=nothing)

One calorically perfect species.

- `name` is the label used in conserved-component names and output.
- `R` is the specific gas constant.
- `gamma` is the constant heat-capacity ratio.

The implied heat capacities are `cv = R / (gamma - 1)` and `cp = cv + R`.
The explicit constructors require `R > 0` and `gamma > 1`. With no `R` and
`gamma`, the name is looked up in the bundled NASA-9 database, `cp` is sampled
at `T_ref`, and `gamma = cp / (cp - R)` forms a calorically perfect
reference-temperature approximation. Use a consistent unit system for `R`,
pressure, density, and temperature.
"""
struct IdealSpecies{T}
    name::String
    R::T          # specific gas constant
    gamma::T

    function IdealSpecies{T}(name::AbstractString, R::Real,
                             gamma::Real) where {T<:AbstractFloat}
        RT, gammaT = T(R), T(gamma)
        isfinite(RT) && RT > zero(T) ||
            throw(ArgumentError("IdealSpecies requires a finite R > 0"))
        isfinite(gammaT) && gammaT > one(T) ||
            throw(ArgumentError("IdealSpecies requires a finite gamma > 1"))
        new{T}(String(name), RT, gammaT)
    end
end

function IdealSpecies(name::AbstractString, R::Real, gamma::Real)
    T = promote_type(typeof(float(R)), typeof(float(gamma)))
    T <: AbstractFloat || (T = Float64)
    return IdealSpecies{T}(name, R, gamma)
end

IdealSpecies(::Type{T}, name::AbstractString, R::Real, gamma::Real) where
    {T<:AbstractFloat} = IdealSpecies{T}(name, R, gamma)

function _ideal_species_from_nasa9_record(::Type{T}, species;
                                          T_ref=298.15) where {T<:AbstractFloat}
    T_ref > 0 || throw(ArgumentError("T_ref must be positive"))
    temperature = T(T_ref)
    cp = species.R * _nasa9_cp_over_R(_nasa9_interval(species, temperature),
                                      temperature)
    cp > species.R || throw(ArgumentError(
        "NASA-9 species $(species.name) has cp <= R at T_ref = $T_ref"))
    return IdealSpecies{T}(species.name, species.R, cp / (cp - species.R))
end

function _ideal_species_from_nasa9(::Type{T}, name::AbstractString;
                                   T_ref=298.15, path=nothing) where
                                   {T<:AbstractFloat}
    database_path = path === nothing ? NASA9_THERMO_PATH : path
    species = read_nasa9(name; path=database_path, reference=:formation,
                         T_ref=T_ref)
    return _ideal_species_from_nasa9_record(T, species; T_ref=T_ref)
end

function IdealSpecies(name::AbstractString; R=nothing, gamma=nothing,
                      T_ref=298.15, path=nothing)
    if R === nothing && gamma === nothing
        return _ideal_species_from_nasa9(Float64, name; T_ref=T_ref, path=path)
    elseif R === nothing || gamma === nothing
        throw(ArgumentError("IdealSpecies requires both R and gamma, or neither"))
    end
    return IdealSpecies(name, R, gamma)
end

function IdealSpecies(::Type{T}, name::AbstractString; R=nothing, gamma=nothing,
                      T_ref=298.15, path=nothing) where {T<:AbstractFloat}
    if R === nothing && gamma === nothing
        return _ideal_species_from_nasa9(T, name; T_ref=T_ref, path=path)
    elseif R === nothing || gamma === nothing
        throw(ArgumentError("IdealSpecies requires both R and gamma, or neither"))
    end
    return IdealSpecies{T}(name, R, gamma)
end

"""
    IdealMixture(species)

Thermally and calorically ideal mixture assembled from nonempty
[`IdealSpecies`](@ref) values. Element types are promoted to a common floating
type. Passing a vector (or tuple) of NASA-9 species names reads them in one
batch and samples each constant heat capacity at `T_ref`. Mixture gas constant
and heat capacities are mass-fraction averages, so each species keeps constant
`R`, `cp`, and `cv` while mixture properties vary with composition.

Vector order defines the order required by every `Prim.Y` tuple, the partial
density components in the conserved state, and species-indexed output and
diagnostics.
"""
struct IdealMixture{T} <: EOS
    sp::Vector{IdealSpecies{T}}
    Rk::Vector{T}
    cvk::Vector{T}
    cpk::Vector{T}
end

function _ideal_mixture(::Type{T}, species) where {T<:AbstractFloat}
    isempty(species) && throw(ArgumentError("IdealMixture requires at least one species"))
    sp = IdealSpecies{T}[IdealSpecies{T}(x.name, x.R, x.gamma) for x in species]
    Rk  = [x.R for x in sp]
    cvk = [x.R / (x.gamma - 1) for x in sp]
    IdealMixture{T}(sp, Rk, cvk, Rk .+ cvk)
end

function IdealMixture(species::AbstractVector{<:IdealSpecies})
    isempty(species) && throw(ArgumentError("IdealMixture requires at least one species"))
    T = promote_type((typeof(x.R) for x in species)...)
    return _ideal_mixture(T, species)
end

IdealMixture(species::Tuple{Vararg{IdealSpecies}}) = IdealMixture(collect(species))
IdealMixture(species::IdealSpecies) = IdealMixture([species])
IdealMixture(::Type{T}, species::AbstractVector{<:IdealSpecies}) where
    {T<:AbstractFloat} = _ideal_mixture(T, species)
IdealMixture(::Type{T}, species::IdealSpecies) where {T<:AbstractFloat} =
    _ideal_mixture(T, (species,))
IdealMixture(::Type{T}, species::Tuple{Vararg{IdealSpecies}}) where
    {T<:AbstractFloat} = _ideal_mixture(T, species)

function IdealMixture(names::AbstractVector{<:AbstractString}; T_ref=298.15,
                      path=nothing)
    database_path = path === nothing ? NASA9_THERMO_PATH : path
    nasa_species = read_nasa9(names; path=database_path, reference=:formation,
                              T_ref=T_ref)
    return IdealMixture([_ideal_species_from_nasa9_record(Float64, sp;
                                                            T_ref=T_ref)
                         for sp in nasa_species])
end

IdealMixture(names::Tuple{Vararg{AbstractString}}; kwargs...) =
    IdealMixture(collect(names); kwargs...)
IdealMixture(::Type{T}, names::Tuple{Vararg{AbstractString}}; kwargs...) where
    {T<:AbstractFloat} = IdealMixture(T, collect(names); kwargs...)
function IdealMixture(::Type{T}, names::AbstractVector{<:AbstractString};
                      T_ref=298.15, path=nothing) where {T<:AbstractFloat}
    database_path = path === nothing ? NASA9_THERMO_PATH : path
    nasa_species = read_nasa9(names; path=database_path, reference=:formation,
                              T_ref=T_ref)
    species = [_ideal_species_from_nasa9_record(T, sp; T_ref=T_ref)
               for sp in nasa_species]
    return IdealMixture(T, species)
end

_default_ideal_mixture() = IdealMixture(IdealSpecies("gas"; R=1.0, gamma=1.4))
_as_eos(eos::EOS) = eos
_as_eos(species::IdealSpecies) = IdealMixture(species)
_as_eos(eos) = throw(ArgumentError(
    "eos must be an EOS or IdealSpecies, got $(typeof(eos))"))

# A literal 1e-300 is a useful nonzero division guard in Float64 but rounds
# to zero in Float32. Preserve the established Float64 value and choose the
# smallest normal value for narrower floating-point types.
@inline positive_floor(::Type{T}) where {T<:AbstractFloat} =
    max(T(1e-300), floatmin(T))
@inline positive_floor(x::T) where {T<:AbstractFloat} = max(x, positive_floor(T))

"""
    nspecies(eos) -> Int

Number of transported species represented by `eos`. This fixes the required
length and ordering of `Prim.Y`, the number of partial-density
components in [`NavierStokes1T`](@ref), and the length of species-indexed
diagnostics.
"""
nspecies(eos::IdealMixture) = length(eos.sp)
species_enthalpy(eos::IdealMixture, k::Int, T_ion) = eos.cpk[k] * T_ion

# φ = cv_m/R_m, recovered from the stored primitives, not from Y: the
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

# The temperature floor of Cook's ρc/T_ion. It is an absolute floor in the
# run's temperature units, not a relative one, and it couples to the
# internal-energy floor in `_primitives_ideal_point!`: a point with e < 0
# gets `T_ion = positive_floor(T)` (1e-300) there, lands on this guard here,
# and contributes ρc/temperature_floor to the diffusive rate, which
# collapses `dt` when a run loses positivity (`stepcontrol.jl`). Nothing in
# the primitives pass signals that it floored; `dt_report` names the cell
# after the fact.
@inline temperature_floor(::Type{T}) where {T<:AbstractFloat} = eps(T)

"Cook's ρc/T_ion. See the note on the singularity at the top of this file."
@inline artificial_conductivity_scale(::IdealMixture, ρ, c, T_ion, cp_mix) =
    ρ * c / max(T_ion, temperature_floor(typeof(T_ion)))

"""
    mixture_cv(eos, ρ, p, T_ion, cp_mix)

Mixture constant-volume heat capacity from the stored primitives, through the
EOS contract: `eos_phi` is cv_m/R_m and cp_m = cv_m + R_m, so
cv_m = cp_m φ/(1 + φ). The explicit thermal-diffusion limit of `compute_dt`
needs κ/(ρ cv), the diffusivity of the internal-energy equation, and this is
the one place it is formed.
"""
@inline function mixture_cv(eos, ρ, p, T_ion, cp_mix)
    φ = eos_phi(eos, ρ, p, T_ion, cp_mix)
    return cp_mix * φ / (1 + φ)
end

# Device-side mirror of the coefficient tables (see pointwise.jl):
# an `IdealMixture` holds `Vector` tables and species names, neither of which
# a kernel argument may carry, so `Adapt.adapt_structure` replaces the whole
# object with this isbits form at kernel launch. Only the methods the
# per-point bodies call exist for it. The tuple construction runs host-side
# once per device launch; the KA CPU backend converts nothing, so the CPU
# paths never see the mirror.
struct IdealMixtureCoeffs{T,N}
    Rk::NTuple{N,T}
    cvk::NTuple{N,T}
    cpk::NTuple{N,T}
end

Adapt.adapt_structure(to, eos::IdealMixture{T}) where {T} =
    IdealMixtureCoeffs{T,length(eos.Rk)}((eos.Rk...,), (eos.cvk...,),
                                         (eos.cpk...,))

Base.@propagate_inbounds species_enthalpy(eos::IdealMixtureCoeffs, k::Int, T_ion) =
    eos.cpk[k] * T_ion
@inline artificial_conductivity_scale(::IdealMixtureCoeffs, ρ, c, T_ion, cp_mix) =
    ρ * c / max(T_ion, temperature_floor(typeof(T_ion)))
@inline function eos_phi(::IdealMixtureCoeffs, ρ, p, T_ion, cp_mix)
    Rm = p / (ρ * T_ion)
    return cp_mix / Rm - 1
end
@inline function eos_dphi_dY(eos::IdealMixtureCoeffs, k::Int, ρ, p, T_ion,
                             cp_mix)
    Rm = p / (ρ * T_ion)
    cvm = cp_mix - Rm
    return (eos.cvk[k] * Rm - eos.Rk[k] * cvm) / (Rm * Rm)
end

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
# and the two derived quantities required by the contract are unusually clean:
# φ = ∂(ρe)/∂p = 1/(γ − 1) exactly, independent of state, and there is no
# composition dependence because this is single-component.
#
# Water at ambient conditions is roughly γ = 4.4, p∞ = 6.0e8 Pa; aluminium and
# other metals are usually done with Mie–Grüneisen instead, which is the same
# shape of model with a density-dependent reference curve and is the natural
# next member of this family.

"""
    StiffenedGas(; gamma=4.4, p_inf=6.0e8, cv=1816.0, name="liquid")

Single-component stiffened-gas equation of state,
`p = (gamma - 1) * rho * e - gamma * p_inf`.

# Keywords

- `gamma`: constant heat-capacity ratio.
- `p_inf`: cohesive pressure that raises the acoustic stiffness.
- `cv`: constant-volume specific heat.
- `name`: species label used in conserved-component names and output.

The thermal relation is `p + p_inf = rho * R * T_ion`, with
`R = (gamma - 1) * cv`. Parameters must use a consistent unit system and be
fitted over the intended material and state range. The defaults are a
water-like starting point in SI units, not a general liquid model. Setting
`p_inf = 0` recovers the perfect-gas algebra of a one-species
[`IdealMixture`](@ref).
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
@inline artificial_conductivity_scale(::StiffenedGas, ρ, c, T_ion, cp_mix) =
    ρ * c / max(T_ion, temperature_floor(typeof(T_ion)))

# `StiffenedGas` carries a name string, so it takes the same launch-time
# mirror treatment as `IdealMixture` above.
struct StiffenedGasCoeffs{T}
    gamma::T
    p_inf::T
    cv::T
end

Adapt.adapt_structure(to, eos::StiffenedGas) =
    StiffenedGasCoeffs(eos.gamma, eos.p_inf, eos.cv)

@inline species_enthalpy(eos::StiffenedGasCoeffs, ::Int, T_ion) =
    eos.gamma * eos.cv * T_ion
@inline artificial_conductivity_scale(::StiffenedGasCoeffs, ρ, c, T_ion, cp_mix) =
    ρ * c / max(T_ion, temperature_floor(typeof(T_ion)))
@inline eos_phi(eos::StiffenedGasCoeffs, ρ, p, T_ion, cp_mix) =
    1 / (eos.gamma - 1)
@inline eos_dphi_dY(::StiffenedGasCoeffs, ::Int, ρ, p, T_ion, cp_mix) = 0.0

@inline function _primitives_stiffened_point!(Q, ρa, ua, va, wa, pa, T_iona,
                                              ca, cpa, Y1, γ, p_inf, cv, R,
                                              m1, m2, m3, i_energy, i, j, k)
    @inbounds begin
        ρ = Q[i, j, k, 1]
        if ρ > 0
            ri = one(ρ) / ρ
            u = Q[i, j, k, m1] * ri
            v = Q[i, j, k, m2] * ri
            w = Q[i, j, k, m3] * ri
            e = Q[i, j, k, i_energy] * ri -
                (u*u + v*v + w*w) / oftype(ρ, 2)
            p = (γ - 1) * ρ * e - γ * p_inf
            ρa[i, j, k] = ρ
            ua[i, j, k] = u; va[i, j, k] = v; wa[i, j, k] = w
            pa[i, j, k] = p
            T_iona[i, j, k] = positive_floor((p + p_inf) / (ρ * R))
            ca[i, j, k] = sqrt(max(γ * (p + p_inf) * ri, zero(ρ)))
            cpa[i, j, k] = γ * cv
            Y1[i, j, k] = 1
        else
            ρa[i, j, k] = 1
            ua[i, j, k] = 0; va[i, j, k] = 0; wa[i, j, k] = 0
            pa[i, j, k] = 1; T_iona[i, j, k] = 1; ca[i, j, k] = 1
            cpa[i, j, k] = γ * cv
            Y1[i, j, k] = 1
        end
    end
    return nothing
end

function recover_primitives!(solver, eos::StiffenedGas, Q)
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    γ = eos.gamma; p_inf = eos.p_inf; cv = eos.cv
    R = gas_constant(eos)
    ρa, ua, va, wa = solver.rho, solver.u, solver.v, solver.w
    pa, T_iona, ca, cpa = solver.p, solver.T_ion, solver.c, solver.cp_mix
    nxf, nyf, nzf = padded_extent(solver.decomp)
    pointwise!(_primitives_stiffened_point!, ρa, nxf, nyf, nzf,
               Q, ρa, ua, va, wa, pa, T_iona, ca, cpa, solver.Y[1],
               γ, p_inf, cv, R, m1, m2, m3, i_energy)
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
# the consistency test checks numerically without trusting the transcription.
#
# The NASA CEA coefficient database is bundled verbatim in `data/thermo.inp`;
# its Apache license and notice live beside it. `read_nasa9` reads the piecewise
# fixed-column records and derives each specific R from the tabulated molar mass,
# keeping coefficients and gas constants coupled. `nasa9_constant_cp` builds the
# degenerate constant-cp case used to pin the machinery against `IdealMixture`.
#
# The cost of a caloric EOS is that T is no longer explicit in e:
# `recover_primitives!`
# runs a Newton iteration per point, converging on de/dT = cv. That is inherent
# to the model, not to this implementation.

"""
    Nasa9Interval(Tmin, Tmax, a, b1)

One temperature interval of a NASA-9 polynomial.

- `Tmin` and `Tmax` are the fit bounds in kelvin and must satisfy
  `0 < Tmin < Tmax`.
- `a` contains the seven coefficients of `cp/R`, from the inverse-square term
  through the fourth-power term.
- `b1` is the integration constant in `h/R`.

All values use the same numeric type. Users normally obtain intervals through
[`read_nasa9`](@ref); direct construction is useful for analytic models and
tests.
"""
struct Nasa9Interval{T}
    Tmin::T
    Tmax::T
    a::NTuple{7,T}
    b1::T
end

@inline function _nasa9_cp_over_R(interval::Nasa9Interval, T_ion)
    a = interval.a
    return a[1] / T_ion^2 + a[2] / T_ion + a[3] + a[4] * T_ion +
           a[5] * T_ion^2 + a[6] * T_ion^3 + a[7] * T_ion^4
end

@inline function _nasa9_h_over_R(interval::Nasa9Interval, T_ion)
    a = interval.a
    return -a[1] / T_ion + a[2] * log(T_ion) + a[3] * T_ion +
           a[4] * T_ion^2 / 2 + a[5] * T_ion^3 / 3 +
           a[6] * T_ion^4 / 4 + a[7] * T_ion^5 / 5 + interval.b1
end

# The join check exists to catch a mistranscribed coefficient, which puts an
# O(1) kink in cp, not to audit the quality of the upstream fit. Sized from the
# bundled CEA table: of 1276 gaseous multi-interval records, the relative cp/R
# jump at a join is ≲1e-8 for all but two (SnCL2 at 1.0e-5, ALOCL at 5.1e-4).
# Rejecting those would be overreach, since 0.05% in cp is far below any
# discretization error here, so the threshold sits above them and well below
# anything a wrong digit could produce.
const NASA9_CONTINUITY_RTOL = 2e-3

function _validate_nasa9_intervals(name, R, intervals)
    isempty(intervals) && throw(ArgumentError("NASA-9 species $name has no intervals"))
    R > 0 || throw(ArgumentError("NASA-9 species $name has non-positive R"))
    for interval in intervals
        interval.Tmin > 0 && interval.Tmin < interval.Tmax ||
            throw(ArgumentError("invalid NASA-9 interval for $name"))
    end
    for i in 1:length(intervals)-1
        lo, hi = intervals[i], intervals[i + 1]
        scale_T = max(abs(lo.Tmax), abs(hi.Tmin), one(lo.Tmax))
        abs(lo.Tmax - hi.Tmin) <= 1e-10 * scale_T ||
            throw(ArgumentError("non-contiguous NASA-9 intervals for $name"))
        T_join = (lo.Tmax + hi.Tmin) / 2
        cp_lo = _nasa9_cp_over_R(lo, T_join)
        cp_hi = _nasa9_cp_over_R(hi, T_join)
        cp_scale = max(abs(cp_lo), abs(cp_hi), one(cp_lo))
        abs(cp_lo - cp_hi) <= NASA9_CONTINUITY_RTOL * cp_scale ||
            throw(ArgumentError("discontinuous NASA-9 cp for $name at $T_join K"))
        h_lo = _nasa9_h_over_R(lo, T_join)
        h_hi = _nasa9_h_over_R(hi, T_join)
        h_scale = max(abs(h_lo), abs(h_hi), T_join)
        abs(h_lo - h_hi) <= NASA9_CONTINUITY_RTOL * h_scale ||
            throw(ArgumentError("discontinuous NASA-9 h for $name at $T_join K"))
    end
    return intervals
end

"""
    Nasa9Species(name, R, intervals)
    Nasa9Species(; name, R, a, b1=0.0, Tmin=200.0, Tmax=6000.0)

One species of a [`Nasa9Mixture`](@ref). `name` is the output label, `R` is the
specific gas constant, and `intervals` is an ordered vector of
[`Nasa9Interval`](@ref) fits.

Construction checks that `R` and temperature bounds are positive, intervals
are contiguous, and `cp` and `h` are continuous at their joins to the tolerance
used for the bundled database. Use [`read_nasa9`](@ref) for CEA data so `R` is
derived from the table's molar mass, not entered independently.

The keyword constructor with `a`, `b1`, `Tmin`, and `Tmax` creates one interval
and defaults to `Float64`. It is intended for small analytic coefficient sets
and tests.
"""
struct Nasa9Species{T}
    name::String
    R::T
    intervals::Vector{Nasa9Interval{T}}

    function Nasa9Species{T}(name::String, R::T,
                             intervals::Vector{Nasa9Interval{T}}) where {T}
        _validate_nasa9_intervals(name, R, intervals)
        new{T}(name, R, intervals)
    end
end

Nasa9Species(name::AbstractString, R::T,
             intervals::Vector{Nasa9Interval{T}}) where {T} =
    Nasa9Species{T}(String(name), R, intervals)

function Nasa9Species{T}(; name, R, a, b1=zero(T), Tmin=T(200),
                           Tmax=T(6000)) where {T}
    coeffs = ntuple(i -> T(a[i]), 7)
    interval = Nasa9Interval{T}(T(Tmin), T(Tmax), coeffs, T(b1))
    return Nasa9Species{T}(String(name), T(R), [interval])
end

Nasa9Species(; kwargs...) = Nasa9Species{Float64}(; kwargs...)

"""
    Nasa9Species(name; path=nothing, reference=:sensible, T_ref=298.15)

Read one gaseous NASA-9 species from the bundled CEA database. This is the
single-species convenience form of [`read_nasa9`](@ref); pass `Float32` (or
another floating type) first to convert the fitted coefficients.
"""
function _convert_nasa9_species(::Type{T}, species) where {T<:AbstractFloat}
    intervals = [Nasa9Interval{T}(T(item.Tmin), T(item.Tmax),
                                  ntuple(i -> T(item.a[i]), 7), T(item.b1))
                 for item in species.intervals]
    return Nasa9Species{T}(species.name, T(species.R), intervals)
end

function Nasa9Species(::Type{T}, name::AbstractString; path=nothing,
                      reference=:sensible, T_ref=298.15) where {T<:AbstractFloat}
    database_path = path === nothing ? NASA9_THERMO_PATH : path
    return _convert_nasa9_species(T, read_nasa9(name; path=database_path,
                                                 reference=reference,
                                                 T_ref=T_ref))
end

Nasa9Species(name::AbstractString; kwargs...) = Nasa9Species(Float64, name; kwargs...)

"""
    nasa9_constant_cp([T=Float64], name, R, cp) -> Nasa9Species

Construct a [`Nasa9Species`](@ref) whose specific heat `cp` is temperature
independent. The default coefficient type is `Float64`; pass `Float32` as the
first argument for a Float32 mechanism. `name` is its output label and `R` its
specific gas constant. This is the calorically perfect limiting case used to
compare NASA-9 machinery with [`IdealSpecies`](@ref).
"""
function nasa9_constant_cp(::Type{T}, name::String, R::Real,
                           cp::Real) where {T<:AbstractFloat}
    z = zero(T)
    return Nasa9Species{T}(name=name, R=T(R),
                           a=(z, z, T(cp / R), z, z, z, z), b1=z)
end

nasa9_constant_cp(name::String, R::Real, cp::Real) =
    nasa9_constant_cp(Float64, name, R, cp)

"""
    Nasa9Mixture(species; T_guess=300.0, extrapolate=:polynomial)

Multicomponent mixture with piecewise NASA-9 heat capacities. `species` is a
nonempty vector of [`Nasa9Species`](@ref) values; vector order defines every
`Prim.Y` tuple, conserved partial density, and species-indexed output.

The mixture is thermally ideal (`p = rho * R_mix * T_ion`) but not calorically:
`cp`, `cv`, and `gamma` vary with temperature. Recovering temperature from
internal energy therefore uses a safeguarded Newton iteration; `mixture_temperature`
documents its convergence criterion and the status it reports. `T_guess` is the
fixed reference temperature used to form a state-based initial estimate; its
default is `300.0` K. It should lie in the representative range of the fits.

`extrapolate` selects what happens outside the union of a species' tabulated
intervals, where nothing in the data constrains the fit:

- `:polynomial` (default) evaluates the nearest interval's fit at the requested
  temperature. This is the conventional treatment and reproduces earlier
  releases exactly, but a degree-four fit run beyond its range can turn `cp`
  negative, which destroys the monotonicity the temperature inversion relies on.
- `:linear` freezes `cp` at the value it takes at the nearest interval endpoint
  and continues `h` linearly from there. The extension is monotone in `T` and
  keeps the inversion bracketed at any temperature, at the cost of a first
  derivative that is discontinuous at the edge of the fitted range.
- `:missing` treats the fit as undefined outside its intervals. The evaluation
  still returns the `:linear` extension, so a step in progress completes and the
  offending state can be inspected, but the point is reported as inadmissible
  and not merely extrapolated, so a run under `StepControl(validity = :strict)`
  fails on it.

This setting, not the state validation, decides whether leaving the fitted range
ends a run. `:polynomial` and `:linear` are both statements that extrapolation
is acceptable for this calculation, and a run using them is not rejected for
extrapolating; the points are still counted and reported. `:missing` is the
statement that it is not acceptable. An extrapolated state is reported under
every setting, since successful evaluation of the polynomial is not evidence
that the state lies in the model's domain.
"""
struct Nasa9Mixture{T} <: EOS
    sp::Vector{Nasa9Species{T}}
    Rk::Vector{T}
    T_guess::T
    extrapolate::Symbol
end

function _nasa9_mixture(::Type{T}, species; T_guess=300.0,
                        extrapolate::Symbol=:polynomial) where {T<:AbstractFloat}
    isempty(species) && throw(ArgumentError("Nasa9Mixture requires at least one species"))
    extrapolate in (:polynomial, :linear, :missing) ||
        throw(ArgumentError("Nasa9Mixture: extrapolate must be :polynomial, " *
                            ":linear or :missing, got :$extrapolate"))
    sp = Nasa9Species{T}[_convert_nasa9_species(T, item) for item in species]
    return Nasa9Mixture{T}(sp, [x.R for x in sp], T(T_guess), extrapolate)
end

function Nasa9Mixture(species::AbstractVector{<:Nasa9Species}; T_guess=300.0,
                      extrapolate::Symbol=:polynomial)
    isempty(species) && throw(ArgumentError("Nasa9Mixture requires at least one species"))
    T = promote_type((typeof(item.R) for item in species)...)
    return _nasa9_mixture(T, species; T_guess=T_guess, extrapolate=extrapolate)
end

Nasa9Mixture(species::Nasa9Species; kwargs...) = Nasa9Mixture([species]; kwargs...)
Nasa9Mixture(species::Tuple{Vararg{Nasa9Species}}; kwargs...) =
    Nasa9Mixture(collect(species); kwargs...)
Nasa9Mixture(::Type{T}, species::AbstractVector{<:Nasa9Species}; kwargs...) where
    {T<:AbstractFloat} = _nasa9_mixture(T, species; kwargs...)
Nasa9Mixture(::Type{T}, species::Nasa9Species; kwargs...) where
    {T<:AbstractFloat} = _nasa9_mixture(T, (species,); kwargs...)
Nasa9Mixture(::Type{T}, species::Tuple{Vararg{Nasa9Species}}; kwargs...) where
    {T<:AbstractFloat} = _nasa9_mixture(T, species; kwargs...)

function Nasa9Mixture(names::AbstractVector{<:AbstractString}; path=nothing,
                      reference=:sensible, T_ref=298.15, T_guess=300.0,
                      extrapolate::Symbol=:polynomial)
    database_path = path === nothing ? NASA9_THERMO_PATH : path
    species = read_nasa9(names; path=database_path, reference=reference,
                         T_ref=T_ref)
    return _nasa9_mixture(Float64, species; T_guess=T_guess,
                          extrapolate=extrapolate)
end

Nasa9Mixture(names::Tuple{Vararg{AbstractString}}; kwargs...) =
    Nasa9Mixture(collect(names); kwargs...)
Nasa9Mixture(::Type{T}, names::Tuple{Vararg{AbstractString}}; kwargs...) where
    {T<:AbstractFloat} = Nasa9Mixture(T, collect(names); kwargs...)
function Nasa9Mixture(::Type{T}, names::AbstractVector{<:AbstractString};
                      path=nothing, reference=:sensible, T_ref=298.15,
                      T_guess=300.0,
                      extrapolate::Symbol=:polynomial) where {T<:AbstractFloat}
    database_path = path === nothing ? NASA9_THERMO_PATH : path
    species = read_nasa9(names; path=database_path, reference=reference,
                         T_ref=T_ref)
    return _nasa9_mixture(T, species; T_guess=T_guess, extrapolate=extrapolate)
end

nspecies(eos::Nasa9Mixture) = length(eos.sp)
species_names(eos::Nasa9Mixture) = [x.name for x in eos.sp]

@inline function _nasa9_interval(species::Nasa9Species, T_ion)
    intervals = species.intervals
    @inbounds for i in 1:length(intervals)-1
        T_ion <= intervals[i].Tmax && return intervals[i]
    end
    return intervals[end]
end

# Outside the union of a species' intervals the fit carries no information, so
# the two extrapolation policies of `Nasa9Mixture` differ only there. Both
# select the nearest interval; `:linear` additionally evaluates the polynomial
# at the interval endpoint and continues h with that constant cp, which is the
# tangent extension of the fit and is monotone in T at any temperature.
@inline _nasa9_linear(eos::Nasa9Mixture) =
    eos.extrapolate === :linear || eos.extrapolate === :missing

@inline function _nasa9_cp_over_R_at(species::Nasa9Species, T_ion, linear::Bool)
    interval = _nasa9_interval(species, T_ion)
    T_fit = linear ? clamp(T_ion, interval.Tmin, interval.Tmax) : T_ion
    return _nasa9_cp_over_R(interval, T_fit)
end

@inline function _nasa9_h_over_R_at(species::Nasa9Species, T_ion, linear::Bool)
    interval = _nasa9_interval(species, T_ion)
    linear || return _nasa9_h_over_R(interval, T_ion)
    T_fit = clamp(T_ion, interval.Tmin, interval.Tmax)
    return _nasa9_h_over_R(interval, T_fit) +
           _nasa9_cp_over_R(interval, T_fit) * (T_ion - T_fit)
end

# Relative tolerance of the temperature inversion, and of the range test that
# reads its result. A state initialized at exactly the lowest fitted temperature
# recovers an ulp or two below it, and calling that extrapolated would report
# every point of such a state.
@inline _nasa9_rtol(::Type{T}) where {T<:AbstractFloat} =
    max(T(1e-14), T(32) * eps(T))

"Whether `T_ion` lies outside the union of a species' fitted intervals."
@inline function _nasa9_out_of_range(species::Nasa9Species, T_ion)
    margin = _nasa9_rtol(typeof(float(T_ion)))
    @inbounds return T_ion < species.intervals[1].Tmin * (1 - margin) ||
                     T_ion > species.intervals[end].Tmax * (1 + margin)
end

"cp_k(T_ion) from the applicable NASA-9 interval."
@inline function species_cp(eos::Nasa9Mixture, k::Int, T_ion)
    species = eos.sp[k]
    return species.R * _nasa9_cp_over_R_at(species, T_ion, _nasa9_linear(eos))
end

"h_k(T_ion), the exact interval-wise integral of `species_cp`."
@inline function species_enthalpy(eos::Nasa9Mixture, k::Int, T_ion)
    species = eos.sp[k]
    return species.R * _nasa9_h_over_R_at(species, T_ion, _nasa9_linear(eos))
end

"e_k(T_ion) = h_k − R_k T_ion."
@inline species_energy(eos::Nasa9Mixture, k::Int, T_ion) =
    species_enthalpy(eos, k, T_ion) - eos.Rk[k] * T_ion

# Search bounds and iteration cap of the NASA-9 temperature inversion. The
# bounds bracket every temperature the fitted intervals cover, with margin on
# both sides, and give the bisection safeguard a starting interval that needs no
# expansion search. The cap is reached only when the root lies outside those
# bounds or the residual is not monotone; a converging Newton iteration from the
# linearized seed uses a handful.
const NASA9_TEMPERATURE_BOUNDS = (1e-3, 1e9)
const NASA9_TEMPERATURE_ITERATIONS = 40

"""
Status flags returned beside a recovered temperature by
`mixture_temperature_status`. `TEMPERATURE_OK` is the zero value; the others are
independent bits, so a single recovery can report more than one.

- `TEMPERATURE_NOT_CONVERGED`: the residual criterion was not met, either
  because the iteration cap was reached or because it stopped early.
- `TEMPERATURE_NO_BRACKET`: the mixture cv was not positive at some iterate, so
  the residual is not monotone in T and the inversion has no bracket. A
  composition with negative mass fractions produces this.
- `TEMPERATURE_OUT_OF_RANGE`: the result lies outside the fitted intervals of a
  species carrying nonzero mass, by more than the tolerance the inversion
  itself converges to, so the value comes from the extrapolation policy rather
  than from the data.
- `TEMPERATURE_AT_BOUND`: the result sits on a search bound, which means the
  root, if there is one, lies outside `NASA9_TEMPERATURE_BOUNDS`.
"""
const TEMPERATURE_OK = 0x00
const TEMPERATURE_NOT_CONVERGED = 0x01
const TEMPERATURE_NO_BRACKET = 0x02
const TEMPERATURE_OUT_OF_RANGE = 0x04
const TEMPERATURE_AT_BOUND = 0x08

"""
    mixture_temperature(eos, e, Yat) -> T_ion

Invert Σ_k Y_k e_k(T) = e for the temperature. `Yat` is a function of the
species index, so the caller can supply mass fractions from a tuple, a vector,
or straight out of the conserved array without materializing anything. This is
the value-only form of `mixture_temperature_status`, which documents the
iteration and reports whether it succeeded.
"""
@inline mixture_temperature(eos::Nasa9Mixture, e, Yat::F) where {F} =
    mixture_temperature_status(eos, e, Yat)[1]

"""
    mixture_temperature_status(eos, e, Yat) -> (T_ion, status)

Invert Σ_k Y_k e_k(T) = e and report what the inversion did, as a combination of
the `TEMPERATURE_OK` flags. The return value is a plain tuple of the temperature
and a `UInt8`, so this may be called from a per-point loop without allocating.

Safeguarded Newton on f(T) = Σ Y_k e_k(T) − e. The derivative is the mixture cv,
which is positive for any physical coefficient set and composition, so f is
increasing and the sign of the residual at each iterate places the root on one
side of it. The iteration maintains that bracket, starting from
`NASA9_TEMPERATURE_BOUNDS`, and replaces any Newton step leaving it with the
geometric mean of the bracket, which halves the exponent range rather than the
interval and so is the appropriate bisection over twelve decades.

Success is the residual criterion |f| ≤ rtol·T·cv, equivalently a Newton step
of at most rtol·T. Writing it through the derivative the iteration already has
avoids inventing an energy scale, which a formation-enthalpy gauge would make
meaningless: there e is dominated by a constant of formation and its magnitude
says nothing about the accuracy of T. `rtol` is 32 eps of the coefficient type,
floored at 1e-14, so the criterion carries to Float32 unchanged in form.

The iteration stops immediately when the mixture cv is not positive. There is
then no bracket and no unique root, so continuing would report convergence to a
meaningless value; the current estimate is returned with
`TEMPERATURE_NO_BRACKET` set instead. Reaching a search bound or leaving the
fitted range is reported the same way and is not by itself a failure: an
extrapolated recovery is a statement about the model's domain, and
[`Nasa9Mixture`](@ref) documents which extension the fit takes there.

The starting point is a pure function of the state. Seeding from the previous
value at the same point generally saves one iteration but makes the result
depend on call history. A state-based seed preserves the bit-for-bit agreement
between serial and decomposed calculations tested by the MPI suite.
"""
@inline function mixture_temperature_status(eos::Nasa9Mixture, e, Yat::F) where {F}
    n = length(eos.sp)
    # First-order inversion about a fixed reference state. This handles any
    # enthalpy gauge; when e_k = cv_k*T it reduces algebraically to e/cv.
    Tnum = typeof(eos.T_guess)
    lo = Tnum(NASA9_TEMPERATURE_BOUNDS[1])
    hi = Tnum(NASA9_TEMPERATURE_BOUNDS[2])
    e0 = zero(Tnum)
    cv0 = zero(Tnum)
    for k in 1:n
        Yk = Yat(k)
        e0 += Yk * species_energy(eos, k, eos.T_guess)
        cv0 += Yk * (species_cp(eos, k, eos.T_guess) - eos.Rk[k])
    end
    T_ion = cv0 > 0 ? clamp(eos.T_guess + (e - e0) / cv0, lo, hi) : eos.T_guess
    rtol = _nasa9_rtol(Tnum)
    converged = false
    bracketed = true
    for _ in 1:NASA9_TEMPERATURE_ITERATIONS
        f = -e; cvm = zero(Tnum)
        for k in 1:n
            Yk = Yat(k)
            f += Yk * species_energy(eos, k, T_ion)
            cvm += Yk * (species_cp(eos, k, T_ion) - eos.Rk[k])
        end
        if !(cvm > 0)
            bracketed = false
            break
        end
        δ = f / cvm
        T_prev = T_ion
        T_ion = T_ion - δ
        # Tested before the safeguard below, so a converging iteration takes the
        # plain Newton sequence and the bracket costs it nothing. A step of at
        # most rtol·T can be smaller than one ulp, which the strict containment
        # test would otherwise reject and replace with a bisection.
        if abs(δ) <= rtol * T_ion
            converged = true
            break
        end
        # Narrow the bracket on the sign of the residual, which places the root
        # because f increases with T wherever the mixture cv is positive.
        f < 0 ? (lo = T_prev) : (hi = T_prev)
        # Outside the bracket the Newton step is not a correction to anything;
        # bisect instead. This also keeps T positive without a separate limiter.
        # A bracket that has collapsed is no longer informative, so the search
        # falls back to the full range rather than to a degenerate midpoint.
        lo < hi || (lo = Tnum(NASA9_TEMPERATURE_BOUNDS[1]);
                    hi = Tnum(NASA9_TEMPERATURE_BOUNDS[2]))
        (lo < T_ion < hi) || (T_ion = sqrt(lo) * sqrt(hi))
        # A step below one ulp of the estimate cannot make further progress,
        # which happens once the bracket has collapsed onto a search bound.
        T_ion == T_prev && break
    end
    status = TEMPERATURE_OK
    converged || (status |= TEMPERATURE_NOT_CONVERGED)
    bracketed || (status |= TEMPERATURE_NO_BRACKET)
    if T_ion <= Tnum(NASA9_TEMPERATURE_BOUNDS[1]) ||
       T_ion >= Tnum(NASA9_TEMPERATURE_BOUNDS[2])
        status |= TEMPERATURE_AT_BOUND
    end
    for k in 1:n
        if Yat(k) != 0 && _nasa9_out_of_range(eos.sp[k], T_ion)
            status |= TEMPERATURE_OUT_OF_RANGE
            break
        end
    end
    return (T_ion, status)
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

@inline artificial_conductivity_scale(::Nasa9Mixture, ρ, c, T_ion, cp_mix) =
    ρ * c / max(T_ion, temperature_floor(typeof(T_ion)))

function recover_primitives!(solver, eos::Nasa9Mixture, Q)
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    Rk = eos.Rk
    ρa, ua, va, wa = solver.rho, solver.u, solver.v, solver.w
    pa, T_iona, ca, cpa = solver.p, solver.T_ion, solver.c, solver.cp_mix
    nxf, nyf, nzf = size(ρa)
    @threaded nxf*nyf*nzf for jk in outer_indices(nyf, nzf)
        j, k = Tuple(jk)
        @inbounds for i in 1:nxf
            Tnum = eltype(Q)
            ρ = zero(Tnum)
            for sp in 1:n_species
                ρ += Q[i, j, k, sp]
            end
            if ρ > 0
                ri = one(Tnum) / ρ
                Rm = zero(Tnum)
                for sp in 1:n_species
                    solver.Y[sp][i, j, k] = Q[i, j, k, sp] * ri
                    Rm += Q[i, j, k, sp] * Rk[sp]
                end
                Rm *= ri   # one scaling, as `_primitives_ideal_point!` does
                u = Q[i, j, k, m1] * ri
                v = Q[i, j, k, m2] * ri
                w = Q[i, j, k, m3] * ri
                e = Q[i, j, k, i_energy] * ri -
                    (u*u + v*v + w*w) / Tnum(2)
                # Mass fractions straight out of Q: the Newton solve takes an
                # accessor, not a vector, so this stays allocation-free.
                T_ion = mixture_temperature(eos, e, sp -> Q[i, j, k, sp] * ri)
                cpm = zero(Tnum)
                for sp in 1:n_species
                    cpm += solver.Y[sp][i, j, k] * species_cp(eos, sp, T_ion)
                end
                cvm = cpm - Rm
                ρa[i, j, k] = ρ
                ua[i, j, k] = u; va[i, j, k] = v; wa[i, j, k] = w
                pa[i, j, k] = ρ * Rm * T_ion
                T_iona[i, j, k] = T_ion
                ca[i, j, k] = sqrt(max((cpm / cvm) * Rm * T_ion, zero(Tnum)))
                cpa[i, j, k] = cpm
            else
                ρa[i, j, k] = 1
                ua[i, j, k] = 0; va[i, j, k] = 0; wa[i, j, k] = 0
                pa[i, j, k] = 1; T_iona[i, j, k] = 1; ca[i, j, k] = 1
                cpa[i, j, k] = 1
                for sp in 1:n_species
                    solver.Y[sp][i, j, k] = sp == 1 ? 1 : 0
                end
            end
        end
    end
    return solver
end

"""
    Transport(; mu0=0.0, Pr=0.7, Sc=0.7)

Constant molecular-transport model.

# Keywords

- `mu0`: dynamic shear viscosity. Its default `0.0` selects inviscid molecular
  transport; artificial properties remain independently controlled by
  [`ArtParams`](@ref).
- `Pr`: Prandtl number. Molecular conductivity is `mu0 * cp / Pr`.
- `Sc`: Schmidt number. Each molecular species diffusivity is
  `mu0 / (rho * Sc)`.

`Pr` and `Sc` are dimensionless and should be positive. The constructor does
not enforce positivity. All three values must use the same numeric type.
"""
Base.@kwdef struct Transport{T}
    mu0::T = 0.0
    Pr::T  = 0.7
    Sc::T  = 0.7
end

"""
    primitives!(solver, Q)

Recover ρ, u, v, w, p, T_ion, c, cp_mix, and the species mass fractions Y_k over
the full padded arrays. The rank-boundary halos of `Q` must be current;
this function performs no communication of its own, and
[`refresh_primitives!`](@ref) is the collective wrapper that exchanges first.

Wherever the partial densities of `Q` do not sum to a positive density, the point
receives finite placeholders (ρ = 1, zero velocity, unit pressure, temperature
and sound speed, all mass on the first species), preventing a division by zero.
A physical-edge halo is the ordinary case, since no exchange reaches it and the
boundary conditions write the edge plane, not the halo behind it. No
compact operator reads those halos either: at a closed edge the closure rows
reference interior points only, and at a fold `fold_fill!` overwrites them from
the interior before the sweep.
"""
primitives!(solver, Q) = recover_primitives!(solver, solver.eos, Q)

@inline function _primitives_ideal_point!(Q, ρa, ua, va, wa, pa, T_iona, ca,
                                          cpa, Y, eos, n_species,
                                          m1, m2, m3, i_energy, i, j, k)
    @inbounds begin
        Rk, cvk = eos.Rk, eos.cvk
        ρ = zero(eltype(Q))
        for sp in 1:n_species
            ρ += Q[i, j, k, sp]
        end
        if ρ > 0
            ri = one(ρ) / ρ
            Rm = zero(ρ); cvm = zero(ρ)
            for sp in 1:n_species
                Rm  += Q[i, j, k, sp] * Rk[sp]
                cvm += Q[i, j, k, sp] * cvk[sp]
                Y[sp][i, j, k] = Q[i, j, k, sp] * ri
            end
            Rm *= ri; cvm *= ri
            u = Q[i, j, k, m1] * ri
            v = Q[i, j, k, m2] * ri
            w = Q[i, j, k, m3] * ri
            e = Q[i, j, k, i_energy] * ri -
                (u*u + v*v + w*w) / oftype(ρ, 2)
            T_ion = positive_floor(e / cvm)
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
                Y[sp][i, j, k] = sp == 1 ? 1 : 0
            end
        end
    end
    return nothing
end

function recover_primitives!(solver, eos::IdealMixture, Q)
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    ρa, ua, va, wa = solver.rho, solver.u, solver.v, solver.w
    pa, T_iona, ca, cpa = solver.p, solver.T_ion, solver.c, solver.cp_mix
    nxf, nyf, nzf = padded_extent(solver.decomp)
    pointwise!(_primitives_ideal_point!, ρa, nxf, nyf, nzf,
               Q, ρa, ua, va, wa, pa, T_iona, ca, cpa,
               solver.field_tuples.Y, eos,
               n_species, m1, m2, m3, i_energy)
    return solver
end

# ---------------------------------------------------------------------------
# Mole fractions, the field the bulk species channel senses beside the mass
# fraction (artificial.jl, `bulk_diffusivity!`). For an ideal mixture at one
# pressure and temperature the mole fraction is also the volume fraction, so
# it sits on the density interface where the mass fraction, weighted by the
# molecular masses, does not.
#
# Σ_j Y_j R_j is the mixture gas constant and passes through zero when a
# light-gas undershoot sits inside a much heavier gas (Y_light = −1/R at density
# ratio R is enough), so the denominator is floored and the result clamped to
# [−1, 2]: the excursion the bound term reads then saturates instead of
# diverging. Both bounds lie outside anything a mass fraction reaches under the
# bound, so the clamp is inert where the state is healthy.
@inline function _mole_fraction_from_R(Rk, k::Int, Y, I, n_species::Int)
    @inbounds begin
        y = Y[k][I]
        s = zero(y)
        for j in 1:n_species
            s += Y[j][I] * Rk[j]
        end
        s = positive_floor(s)
        return clamp(y * Rk[k] / s, -one(y), 2 * one(y))
    end
end

"""
    mole_fraction(eos, k, Y, I, n_species)

Mole fraction of species `k` at padded index `I`, from the mass fractions `Y`
(a `FieldVector` of the padded arrays). For the gas mixtures it is
Y_k R_k / Σ_j Y_j R_j with the denominator floored at `positive_floor` and the
result clamped to [−1, 2]; a single-material EOS returns the mass fraction.
Part of the EOS contract (see the header); consumed by the bulk species
channel of `compute_artificial!`.
"""
@inline mole_fraction(eos::IdealMixture, k::Int, Y, I, n_species::Int) =
    _mole_fraction_from_R(eos.Rk, k, Y, I, n_species)
@inline mole_fraction(eos::IdealMixtureCoeffs, k::Int, Y, I, n_species::Int) =
    _mole_fraction_from_R(eos.Rk, k, Y, I, n_species)
@inline mole_fraction(eos::Nasa9Mixture, k::Int, Y, I, n_species::Int) =
    _mole_fraction_from_R(eos.Rk, k, Y, I, n_species)
Base.@propagate_inbounds mole_fraction(::StiffenedGas, k::Int, Y, I, ::Int) =
    Y[k][I]

# ---------------------------------------------------------------------------
# Admissibility: whether a point lies in the thermodynamic domain of its EOS.
#
# The flags below describe one point of a conserved state. The first three are
# properties of the numbers themselves and are the same for every model; the
# last three are the EOS's own statement about the state and are produced by
# `state_admissibility`.
#
# There is deliberately no universal e > 0 test. Internal energy is defined only
# up to the gauge its enthalpy reference fixes, so e < 0 is a failure for a
# calorically perfect gas, where e = cv T, and the ordinary case for a NASA-9
# mixture in the formation gauge, where the heats of formation of the species
# set the zero. Asking the EOS keeps that distinction with the model that owns
# it.

"""
Per-point validity flags of a conserved state. `STATE_OK` is the zero value and
the rest are independent bits, so one point can carry several.

- `STATE_NONFINITE`: a conserved component is not finite.
- `STATE_NEGATIVE_DENSITY`: the partial densities do not sum to a positive
  mixture density, which is the state `primitives!` replaces with placeholders.
- `STATE_NEGATIVE_SPECIES`: a partial density is negative while the mixture
  density is positive, by more than the dead band `ArtParams.Y_tolerance`
  places on the artificial mass-fraction bound. A filtered interface carries
  mass fractions a rounding step below zero over much of a domain, which that
  band exists to ignore.
- `STATE_INADMISSIBLE`: the EOS places the point outside its thermodynamic
  domain.
- `STATE_UNRECOVERABLE`: the EOS could not recover the temperature at this
  point. Only a model that inverts a caloric relation can report it.
- `STATE_EXTRAPOLATED`: the EOS evaluated outside the range its data covers, so
  the recovered state rests on the extrapolation policy rather than on the fit.
  Informational on its own: whether it is also a failure is the model's to say,
  and a model that says so sets `STATE_INADMISSIBLE` beside it.
"""
const STATE_OK = 0x00
const STATE_NONFINITE = 0x01
const STATE_NEGATIVE_DENSITY = 0x02
const STATE_NEGATIVE_SPECIES = 0x04
const STATE_INADMISSIBLE = 0x08
const STATE_UNRECOVERABLE = 0x10
const STATE_EXTRAPOLATED = 0x20

"""
    state_admissibility(eos, rho, e, Yat, n_species) -> UInt8

Flags describing whether the point `(rho, e, Y)` lies in the thermodynamic
domain of `eos`, where `e` is specific internal energy and `Yat` is a function
of the species index returning a mass fraction, as `mixture_temperature` takes.
`STATE_OK` means admissible; the value is otherwise a combination of
`STATE_INADMISSIBLE`, `STATE_UNRECOVERABLE`, and `STATE_EXTRAPOLATED`.

Part of the [`EOS`](@ref) contract. A model that cannot decide should return
`STATE_OK`, which leaves the finiteness and density tests of the state
validation as the only checks applied to it. Implementations must be free of
allocation, since this runs per point.
"""
function state_admissibility end

# A calorically perfect mixture puts zero internal energy at zero temperature,
# so T = e/cv_m is positive exactly when e and the mixture cv are. The pressure
# and sound speed follow from T and are then positive as well.
@inline function state_admissibility(eos::IdealMixture, ρ, e, Yat::F,
                                     n_species::Int) where {F}
    cvm = zero(e)
    for k in 1:n_species
        cvm += Yat(k) * eos.cvk[k]
    end
    return (cvm > 0 && e > 0) ? STATE_OK : STATE_INADMISSIBLE
end

# ρe = ρ c_v T + p∞, so the thermal part of the internal energy is e − p∞/ρ and
# it, not e, carries the sign of the temperature.
@inline state_admissibility(eos::StiffenedGas, ρ, e, Yat::F, n_species::Int) where {F} =
    (ρ > 0 && e - eos.p_inf / ρ > 0) ? STATE_OK : STATE_INADMISSIBLE

# The caloric inversion is the admissibility test: a state it cannot invert is
# one this model does not describe, and one it inverts outside the fitted range
# is described by the extrapolation policy rather than by the data. Whether that
# policy makes the state a failure is the mixture's `extrapolate` setting, not
# the validation's: `:polynomial` and `:linear` accept the extension and report
# the point, `:missing` declares the fit undefined there.
@inline function state_admissibility(eos::Nasa9Mixture, ρ, e, Yat::F,
                                     n_species::Int) where {F}
    _, status = mixture_temperature_status(eos, e, Yat)
    flags = STATE_OK
    if (status & (TEMPERATURE_NOT_CONVERGED | TEMPERATURE_NO_BRACKET)) != 0
        flags |= STATE_UNRECOVERABLE
    end
    if (status & TEMPERATURE_OUT_OF_RANGE) != 0
        flags |= STATE_EXTRAPOLATED
        eos.extrapolate === :missing && (flags |= STATE_INADMISSIBLE)
    end
    return flags
end
