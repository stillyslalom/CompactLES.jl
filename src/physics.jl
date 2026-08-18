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
# A new EOS supplies these. `eos_phi`, `eos_dphi_dY` and `art_conductivity_scale`
# existed as ideal-gas algebra inlined at their call sites, which left the rest of
# the solver quietly ideal-gas even though this file was not; the others have been
# dispatch points throughout.
#
#   nspecies(eos)                        number of species Ns
#   _primitives!(solver, eos, Q)         ρ, u, v, w, p, T_ion, c, cp_mix, Y over
#                                        the padded arrays — the bulk conversion
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
# One further method is optional: `species_names(eos)` labels the partial-density
# components, and the fallback in equations.jl returns "species_1", "species_2"
# and so on for an EOS that does not define it.
#
# Several of these are evaluated per point: `species_enthalpy` in
# `_assemble_fluxes!`, `art_conductivity_scale` in `compute_artificial!`, and
# `wall_internal_energy` over an isothermal wall plane. Each such loop sits inside
# a routine specialized on the concrete solver or EOS type, so the dispatch on
# `eos` is paid once per array pass or once per boundary plane, not per point.
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
    IdealSpecies(name, R, gamma)

One calorically perfect species.

- `name` is the label used in conserved-component names and output.
- `R` is the specific gas constant.
- `gamma` is the constant heat-capacity ratio.

The implied heat capacities are `cv = R / (gamma - 1)` and `cp = cv + R`.
The constructor does not validate the parameters; physical use requires
`R > 0` and `gamma > 1`. Use a consistent unit system for `R`, pressure,
density, and temperature.
"""
struct IdealSpecies{T}
    name::String
    R::T          # specific gas constant
    gamma::T
end

"""
    IdealMixture(species)

Thermally and calorically ideal mixture assembled from a nonempty vector of
[`IdealSpecies`](@ref) values with a common numeric type. Mixture gas constant
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

function IdealMixture(sp::Vector{IdealSpecies{T}}) where {T}
    Rk  = [x.R for x in sp]
    cvk = [x.R / (x.gamma - 1) for x in sp]
    IdealMixture{T}(sp, Rk, cvk, Rk .+ cvk)
end

"""
    single_species(; gamma=1.4, R=1.0, name="gas") -> IdealMixture

Construct a one-species calorically perfect [`IdealMixture`](@ref). `gamma` is
the heat-capacity ratio, `R` the specific gas constant, and `name` the label
used for the partial-density component and output. This is the default EOS for
[`Problem`](@ref).
"""
single_species(; gamma::Real=1.4, R::Real=1.0, name::String="gas") =
    IdealMixture([IdealSpecies{Float64}(name, Float64(R), Float64(gamma))])

"""
    nspecies(eos) -> Int

Number of transported species represented by `eos`. This fixes the required
length and ordering of `Prim.Y`, the number of partial-density
components in [`NavierStokes1T`](@ref), and the length of species-indexed
diagnostics.
"""
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

# Device-side mirror of the coefficient tables (GPU Stage G1, pointwise.jl):
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
@inline art_conductivity_scale(::IdealMixtureCoeffs, ρ, c, T_ion, cp_mix) =
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
`p_inf = 0` recovers the perfect-gas algebra of [`single_species`](@ref).
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

# `StiffenedGas` carries a name string, so it takes the same launch-time
# mirror treatment as `IdealMixture` above.
struct StiffenedGasCoeffs{T}
    gamma::T
    cv::T
end

Adapt.adapt_structure(to, eos::StiffenedGas) =
    StiffenedGasCoeffs(eos.gamma, eos.cv)

@inline species_enthalpy(eos::StiffenedGasCoeffs, ::Int, T_ion) =
    eos.gamma * eos.cv * T_ion
@inline art_conductivity_scale(::StiffenedGasCoeffs, ρ, c, T_ion, cp_mix) =
    ρ * c / max(T_ion, eps(typeof(T_ion)))

@inline function _primitives_stiffened_point!(Q, ρa, ua, va, wa, pa, T_iona,
                                              ca, cpa, Y1, γ, p_inf, cv, R,
                                              m1, m2, m3, i_energy, i, j, k)
    @inbounds begin
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
            Y1[i, j, k] = 1.0
        else
            ρa[i, j, k] = 1
            ua[i, j, k] = 0; va[i, j, k] = 0; wa[i, j, k] = 0
            pa[i, j, k] = 1; T_iona[i, j, k] = 1; ca[i, j, k] = 1
            cpa[i, j, k] = γ * cv
            Y1[i, j, k] = 1.0
        end
    end
    return nothing
end

function _primitives!(solver, eos::StiffenedGas, Q)
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    γ = eos.gamma; p_inf = eos.p_inf; cv = eos.cv
    R = gas_constant(eos)
    ρa, ua, va, wa = solver.rho, solver.u, solver.v, solver.w
    pa, T_iona, ca, cpa = solver.p, solver.T_ion, solver.c, solver.cp_mix
    nxf, nyf, nzf = size(ρa)
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
# the consistency test checks numerically rather than trusting the transcription.
#
# The NASA CEA coefficient database is shipped verbatim in `data/thermo.inp`;
# its Apache license and notice live beside it. `read_nasa9` reads the piecewise
# fixed-column records and derives each specific R from the tabulated molar mass,
# keeping coefficients and gas constants coupled. `nasa9_constant_cp` builds the
# degenerate constant-cp case used to pin the machinery against `IdealMixture`.
#
# The cost of a caloric EOS is that T is no longer explicit in e: `_primitives!`
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
# O(1) kink in cp — not to audit the quality of the upstream fit. Sized from the
# shipped CEA table: of 1276 gaseous multi-interval records, the relative cp/R
# jump at a join is ≲1e-8 for all but two (SnCL2 at 1.0e-5, ALOCL at 5.1e-4).
# Rejecting those would be the check overreaching — 0.05% in cp is far below any
# discretization error here — so the threshold sits above them and well below
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
derived from the table's molar mass rather than entered independently.

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
    nasa9_constant_cp(name, R, cp) -> Nasa9Species

Construct a `Float64` [`Nasa9Species`](@ref) whose specific heat `cp` is
temperature independent. `name` is its output label and `R` its specific gas
constant. This is the calorically perfect limiting case used to compare
NASA-9 machinery with [`IdealSpecies`](@ref).
"""
nasa9_constant_cp(name::String, R::Real, cp::Real) =
    Nasa9Species{Float64}(name=name, R=Float64(R),
                          a=(0.0, 0.0, Float64(cp / R), 0.0, 0.0, 0.0, 0.0),
                          b1=0.0)

"""
    Nasa9Mixture(species; T_guess=300.0)

Multicomponent mixture with piecewise NASA-9 heat capacities. `species` is a
nonempty vector of [`Nasa9Species`](@ref) values; vector order defines every
`Prim.Y` tuple, conserved partial density, and species-indexed output.

The mixture is thermally ideal (`p = rho * R_mix * T_ion`) but not calorically:
`cp`, `cv`, and `gamma` vary with temperature. Recovering temperature from
internal energy therefore uses a bounded Newton iteration. `T_guess` is the
fixed reference temperature used to form a state-based initial estimate; its
default is `300.0` K. It should lie in the representative range of the fits.

Temperatures outside a species' tabulated range use its first or last
polynomial interval, so successful evaluation is not evidence that such an
extrapolated state is physically valid.
"""
struct Nasa9Mixture{T} <: EOS
    sp::Vector{Nasa9Species{T}}
    Rk::Vector{T}
    T_guess::T
end

Nasa9Mixture(sp::Vector{Nasa9Species{T}}; T_guess=300.0) where {T} =
    Nasa9Mixture{T}(sp, [x.R for x in sp], T(T_guess))

nspecies(eos::Nasa9Mixture) = length(eos.sp)
species_names(eos::Nasa9Mixture) = [x.name for x in eos.sp]

@inline function _nasa9_interval(species::Nasa9Species, T_ion)
    intervals = species.intervals
    @inbounds for i in 1:length(intervals)-1
        T_ion <= intervals[i].Tmax && return intervals[i]
    end
    return intervals[end]
end

"cp_k(T_ion) from the applicable NASA-9 interval."
@inline function species_cp(eos::Nasa9Mixture, k::Int, T_ion)
    species = eos.sp[k]
    return species.R * _nasa9_cp_over_R(_nasa9_interval(species, T_ion), T_ion)
end

"h_k(T_ion), the exact interval-wise integral of `species_cp`."
@inline function species_enthalpy(eos::Nasa9Mixture, k::Int, T_ion)
    species = eos.sp[k]
    return species.R * _nasa9_h_over_R(_nasa9_interval(species, T_ion), T_ion)
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
step is clamped to remain in T > 0 even when a stale halo contains a nonphysical
state, the initial estimate is clamped to [1e-3, 1e9], and the iteration count is
capped at 30. The iteration also stops early if the mixture cv is not positive,
which returns the current estimate rather than raising.

The starting point is a pure function of the state. Seeding from the previous
value at the same point generally saves one iteration but makes the result
depend on call history. A state-based seed preserves the bit-for-bit agreement
between serial and decomposed calculations tested by the MPI suite.
"""
@inline function mixture_temperature(eos::Nasa9Mixture, e, Yat::F) where {F}
    n = length(eos.sp)
    # First-order inversion about a fixed reference state. This handles any
    # enthalpy gauge; when e_k = cv_k*T it reduces algebraically to e/cv.
    e0 = 0.0
    cv0 = 0.0
    for k in 1:n
        Yk = Yat(k)
        e0 += Yk * species_energy(eos, k, eos.T_guess)
        cv0 += Yk * (species_cp(eos, k, eos.T_guess) - eos.Rk[k])
    end
    T_ion = cv0 > 0 ? clamp(eos.T_guess + (e - e0) / cv0, 1e-3, 1e9) :
            eos.T_guess
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
    @threaded nxf*nyf*nzf for jk in outer_indices(nyf, nzf)
        j, k = Tuple(jk)
        @inbounds for i in 1:nxf
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
the full padded arrays. The rank-boundary halos of `Q` must already be current;
this function performs no communication of its own, and
[`refresh_primitives!`](@ref) is the collective wrapper that exchanges first.

Wherever the partial densities of `Q` do not sum to a positive density, the point
receives finite placeholders (ρ = 1, zero velocity, unit pressure, temperature
and sound speed, all mass on the first species) instead of a division by zero.
A physical-edge halo is the ordinary case, since no exchange reaches it and the
boundary conditions write the edge plane rather than the halo behind it. No
compact operator reads those halos either: at a closed edge the closure rows
reference interior points only, and at a fold `fold_fill!` overwrites them from
the interior before the sweep.
"""
primitives!(solver, Q) = _primitives!(solver, solver.eos, Q)

@inline function _primitives_ideal_point!(Q, ρa, ua, va, wa, pa, T_iona, ca,
                                          cpa, Y, eos, n_species,
                                          m1, m2, m3, i_energy, i, j, k)
    @inbounds begin
        Rk, cvk = eos.Rk, eos.cvk
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
                Y[sp][i, j, k] = Q[i, j, k, sp] * ri
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
                Y[sp][i, j, k] = sp == 1 ? 1.0 : 0.0
            end
        end
    end
    return nothing
end

function _primitives!(solver, eos::IdealMixture, Q)
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    ρa, ua, va, wa = solver.rho, solver.u, solver.v, solver.w
    pa, T_iona, ca, cpa = solver.p, solver.T_ion, solver.c, solver.cp_mix
    nxf, nyf, nzf = size(ρa)
    pointwise!(_primitives_ideal_point!, ρa, nxf, nyf, nzf,
               Q, ρa, ua, va, wa, pa, T_iona, ca, cpa,
               solver.field_tuples.Y, eos,
               n_species, m1, m2, m3, i_energy)
    return solver
end
