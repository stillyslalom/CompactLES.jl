# Fully ionized isotope interdiffusion from Stanton and Murillo,
# Phys. Rev. E 93, 043203 (2016), Eqs. 25, 34--36, 65--66, and Appendix C.
# This is a standalone reference-limit evaluator, not a solver transport model.
# NIST CODATA 2022, https://physics.nist.gov/cuu/Constants/Table/allascii.txt
# These are bare-nucleus masses, not neutral atomic masses.
"""Bare H+ nuclear mass in kg (NIST CODATA 2022)."""
const H_ION_MASS = 1.67262192595e-27
"""Bare D+ nuclear mass in kg (NIST CODATA 2022)."""
const D_ION_MASS = 3.3435837768e-27
"""Bare T+ nuclear mass in kg (NIST CODATA 2022)."""
const T_ION_MASS = 5.0073567512e-27
const _SM_ELECTRON_MASS = 9.1093837139e-31
const _SM_EPSILON_0 = 8.8541878188e-12
const _SM_BOLTZMANN = 1.380649e-23
const _SM_ELEMENTARY_CHARGE = 1.602176634e-19
const _SM_HBAR = 6.62607015e-34 / (2pi)

"""
    StantonMurilloDiagnostics

Diagnostics returned by [`stanton_murillo_interdiffusivity`](@ref). All
quantities are SI. `electron_screening_energy` is the finite-temperature
Stanton--Murillo electron-screening energy; `theta = k_B*T/E_F` diagnoses the
electron regime. `ion_quantum_parameter` is ion number density times the cube
of `hbar / sqrt(m_i*k_B*T)` and is diagnostic only.
`assumptions` is always `:fully_ionized_equilibrium`.
"""
struct StantonMurilloDiagnostics{T,N}
    n::T
    ne::T
    electron_screening_energy::T
    Fermi_energy::T
    lambda_e::T
    lambda_i::NTuple{N,T}
    ion_sphere_radius::NTuple{N,T}
    gamma::NTuple{N,T}
    ion_quantum_parameter::NTuple{N,T}
    lambda_eff::T
    coupling::T
    K11::T
    theta::T
    assumptions::Symbol
end

@inline function _sm_K11(g)
    if g < one(g)
        polynomial = muladd(g, muladd(g, muladd(g, muladd(g,
            oftype(g, 0.061162), oftype(g, -0.55833)), oftype(g, 1.4313)),
            oftype(g, -1.7836)), oftype(g, 1.4660))
        return -log(g * polynomial) / 4
    end
    logg = log(g)
    return (oftype(g, 0.081033) + logg * (oftype(g, -0.091336) +
            oftype(g, 0.051760) * logg)) /
           (one(g) + g * (oftype(g, -0.50026) + oftype(g, 0.17044) * g))
end

"""
    stanton_murillo_interdiffusivity(masses, number_fractions, rho,
                                     temperature, i, j;
                                     assume_fully_ionized=false)

Binary ion interdiffusivity for the tested hot, weakly degenerate, fully ionized,
unmagnetized H+/D+/T+ limit of Stanton--Murillo (2016). `masses` are bare
nuclear masses in kg, `number_fractions` are nonnegative and normalized,
`rho` is ion mass density in kg/m^3 (electron mass neglected), and `temperature` is the common
ion/electron equilibrium temperature in K. It returns `(D, diagnostics)`,
where `D` is in m^2/s.

All ions are assumed to have charge state Z=1. This evaluator has no
partial-ionization, cold-start, separate-T, magnetization, radiation, or
ambipolar-field closure and is never selected by the solver. Calling it
requires the explicit `assume_fully_ionized=true`. The `theta >= 10` acceptance
guard defines this tested hot reference subset; it is not a phase boundary or
a neutral-to-ion model switch. Pair coefficients for N > 2 do not define a
complete multicomponent diffusion-flux closure.
"""
function stanton_murillo_interdiffusivity(masses::NTuple{N}, number_fractions::NTuple{N},
                                          rho, temperature, i::Integer, j::Integer;
                                          assume_fully_ionized::Bool=false) where {N}
    assume_fully_ionized || throw(ArgumentError(
        "Stanton--Murillo reference limit requires assume_fully_ionized=true"))
    N > 1 || throw(ArgumentError("Stanton--Murillo interdiffusion requires at least two ion species"))
    1 <= i <= N && 1 <= j <= N || throw(ArgumentError("ion species indices must lie in 1:$N"))
    i != j || throw(ArgumentError("interdiffusion requires distinct ion species"))
    # SI masses and Coulomb factors underflow in Float32; use Float64 for this
    # standalone reference calculation even when its inputs are Float32.
    T = promote_type(Float64, map(typeof, masses)..., map(typeof, number_fractions)...,
                     typeof(float(rho)), typeof(float(temperature)))
    isfinite(rho) && rho > 0 || throw(ArgumentError("mass density must be finite and positive"))
    isfinite(temperature) && temperature > 0 || throw(ArgumentError("temperature must be finite and positive"))
    sum_input = sum(number_fractions)
    isfinite(sum_input) && isapprox(sum_input, one(sum_input);
        rtol=16eps(float(sum_input)), atol=zero(float(sum_input))) ||
        throw(ArgumentError("number fractions must sum to one at their input precision"))
    m = ntuple(k -> T(masses[k]), Val(N))
    x = ntuple(k -> T(number_fractions[k]) / T(sum_input), Val(N))
    all(value -> isfinite(value) && value > zero(T), m) ||
        throw(ArgumentError("ion masses must be finite and positive"))
    all(value -> isfinite(value) && value >= zero(T), x) ||
        throw(ArgumentError("number fractions must be finite and nonnegative"))
    rho_T, temperature_T = T(rho), T(temperature)
    isfinite(rho_T) && rho_T > zero(T) && isfinite(temperature_T) && temperature_T > zero(T) ||
        throw(ArgumentError("density or temperature is not representable as finite and positive"))
    n = rho_T / sum(ntuple(k -> x[k] * m[k], Val(N)))
    ne = n # Z_i = 1 for every ion in this bounded model.
    q2 = T(_SM_ELEMENTARY_CHARGE)^2 / (4T(pi) * T(_SM_EPSILON_0))
    tau = T(_SM_BOLTZMANN) * temperature_T
    Fermi_energy = T(_SM_HBAR)^2 * (3T(pi)^2 * ne)^(T(2) / 3) / (2T(_SM_ELECTRON_MASS))
    isfinite(Fermi_energy) && Fermi_energy > zero(T) ||
        throw(ArgumentError("electron Fermi energy is not finite and positive"))
    electron_screening_energy = hypot(tau, 2Fermi_energy / 3)
    lambda_e = sqrt(T(_SM_EPSILON_0) * electron_screening_energy /
                    (ne * T(_SM_ELEMENTARY_CHARGE)^2))
    lambda_i = ntuple(k -> sqrt(T(_SM_EPSILON_0) * tau /
                                     (x[k] * n * T(_SM_ELEMENTARY_CHARGE)^2)), Val(N))
    radius = ntuple(k -> (3 / (4T(pi) * ne))^(one(T) / 3), Val(N))
    gamma = ntuple(k -> q2 / (radius[k] * tau), Val(N))
    ion_quantum_parameter = ntuple(k -> x[k] * n *
        (T(_SM_HBAR) / sqrt(m[k] * tau))^3, Val(N))
    inverse_lambda_eff_squared = inv(lambda_e)^2 +
        sum(ntuple(k -> inv(lambda_i[k])^2 / (one(T) + 3gamma[k]), Val(N)))
    lambda_eff = inv(sqrt(inverse_lambda_eff_squared))
    coupling = q2 / (lambda_eff * tau)
    K11 = _sm_K11(coupling)
    isfinite(K11) && K11 > zero(T) || throw(ArgumentError("Stanton--Murillo collision integral is not finite and positive"))
    reduced_mass = m[i] * m[j] / (m[i] + m[j])
    D = 3 * tau^(T(5) / 2) / (16sqrt(2T(pi) * reduced_mass) * n * q2^2 * K11)
    isfinite(D) && D > zero(T) || throw(ArgumentError("Stanton--Murillo diffusivity is not finite and positive"))
    theta = tau / Fermi_energy
    isfinite(theta) && theta >= T(10) || throw(ArgumentError(
        "Stanton--Murillo tested hot subset requires theta >= 10; use an EOS/charge-state transport model"))
    diagnostics = StantonMurilloDiagnostics(n, ne, electron_screening_energy, Fermi_energy,
                                             lambda_e, lambda_i, radius, gamma,
                                             ion_quantum_parameter,
                                             lambda_eff, coupling, K11, theta,
                                             :fully_ionized_equilibrium)
    return (D=D, diagnostics=diagnostics)
end
