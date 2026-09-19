# Dilute neutral-gas binary diffusion correlations from T. R. Marrero and
# E. A. Mason, "Gaseous Diffusion Coefficients", J. Phys. Chem. Ref. Data 1,
# 3--118 (1972), https://doi.org/10.1063/1.3253094. The rows below are Tables
# 12 and 13 of that paper, transcribed independently twice from the NIST
# reprint scan and diffed. The row constructors take the printed digits, so
# a row can be checked against the page, and store SI: the printed A is in
# atm cm^2 s^-1 K^-s and becomes Pa m^2 s^-1 K^-s. Every value refers to an
# equimolar mixture except the air pairs, which are trace diffusion through
# air (section 5.2). The paper contains no other isotopic pair: its section
# 5.2 excludes them because self-diffusion is proportional to viscosity, so
# H2-HD and HD-D2 need a separate measured or calculated source.

"""
    MarreroMasonPair

One gas pair of Marrero and Mason's Tables 12 and 13, in SI units.
Equation (4.3-1) of the paper correlates the pressure-diffusivity product as

```math
\\ln(p D_{12}) = \\ln A + s \\ln T - \\ln\\,[\\ln(\\varphi_0/kT)]^2 - S/T - S'/T^2,
```

with ``T`` in K; equation (4.3-2) drops the double-logarithm term and
``S'``. A row from Table 13 stores `phi0_over_k = 0` and `S_prime = 0` and
is evaluated by (4.3-2). `A` is stored in Pa m² s⁻¹ K⁻ˢ, converted from the
paper's atm cm² s⁻¹ K⁻ˢ (the tables print ``10^3 A`` or ``10^5 A``);
`phi0_over_k` is in K (the table prints ``10^{-8}\\varphi_0/k``), `S` in K
and `S_prime` in K². `temperature_min` and `temperature_max` are the
correlation's stated range, and `group` is the paper's reliability group
(`:I`, `:II`, `:III` or `:miscellaneous`), whose uncertainty limits are in
[`MARRERO_MASON_UNCERTAINTY`](@ref). `species` names the pair exactly as
the paper does; `"air"` is a species name. Three pairs (3He-4He, O2-H2O and
air-H2O) carry two rows with different ranges.
"""
struct MarreroMasonPair
    species::NTuple{2,String}
    A::Float64
    s::Float64
    phi0_over_k::Float64
    S::Float64
    S_prime::Float64
    temperature_min::Float64
    temperature_max::Float64
    group::Symbol
end

const _ATMOSPHERE = 101325.0
# atm cm^2/s to Pa m^2/s.
const _ATM_CM2_TO_PA_M2 = _ATMOSPHERE * 1e-4

# Table 12 rows as printed: (10^3 A, s, 10^-8 phi0/k, S, S', Tmin, Tmax, group).
function _marrero_mason_12(a, b, A3, s, phi8, S, S_prime, Tmin, Tmax, group)
    return MarreroMasonPair((a, b), A3 * 1e-3 * _ATM_CM2_TO_PA_M2, s, phi8 * 1e8, S,
                            S_prime, Tmin, Tmax, group)
end

# Table 13 rows as printed: (10^5 A, s, S, Tmin, Tmax, group); a blank is 0.
function _marrero_mason_13(a, b, A5, s, S, Tmin, Tmax, group)
    return MarreroMasonPair((a, b), A5 * 1e-5 * _ATM_CM2_TO_PA_M2, s, 0.0, S, 0.0, Tmin,
                            Tmax, group)
end

"""
    MARRERO_MASON_1972

The correlated gas pairs of Marrero and Mason (1972), Tables 12 and 13, as
[`MarreroMasonPair`](@ref) rows. Look a pair up with
[`marrero_mason_pair`](@ref) and evaluate it with
[`marrero_mason_diffusivity`](@ref); [`neutral_binary_diffusion`](@ref)
fits a [`BinaryDiffusionPolynomial`](@ref) to the rows of a species list.
The only hydrogen isotopologue pair is H2-D2. The paper's N2 and CO rows
share coefficients by design, since it treats the two as isosteric.
"""
const MARRERO_MASON_1972 = MarreroMasonPair[
    # Table 12, eq (4.3-1)
    _marrero_mason_12("3He", "4He", 32.4, 1.501, 0.0448, -0.963, 1.894, 1.74, 1e4, :II),
    _marrero_mason_12("3He", "4He", 0.156, 1.636, 0.0, 0.0, 0.0, 14.4, 90.0, :II),
    _marrero_mason_12("He", "Ne", 25.41, 1.509, 0.212, 1.87, 0.0, 65.0, 1e4, :I),
    _marrero_mason_12("He", "Ar", 15.21, 1.552, 0.41, 1.71, 0.0, 77.0, 1e4, :I),
    _marrero_mason_12("He", "Kr", 10.61, 1.609, 1.42, -32.65, 2036.0, 77.0, 1e4, :I),
    _marrero_mason_12("He", "Xe", 7.981, 1.644, 4.02, -68.87, 5416.0, 169.0, 1e4, :I),
    _marrero_mason_12("He", "H2", 27.0, 1.51, 0.0534, 0.0, 0.0, 90.0, 1e4, :II),
    _marrero_mason_12("He", "N2", 15.8, 1.524, 0.265, 0.0, 0.0, 77.0, 1e4, :II),
    _marrero_mason_12("He", "CO", 15.8, 1.524, 0.265, 0.0, 0.0, 77.0, 1e4, :II),
    _marrero_mason_12("Ne", "Ar", 8.779, 1.546, 1.94, 1.82, 1170.0, 90.0, 1e4, :I),
    _marrero_mason_12("Ne", "Kr", 8.52, 1.555, 6.73, 20.4, 0.0, 112.0, 1e4, :I),
    _marrero_mason_12("Ne", "Xe", 6.747, 1.584, 19.0, 10.1, 0.0, 169.0, 1e4, :I),
    _marrero_mason_12("Ar", "Kr", 5.346, 1.556, 13.0, 47.3, 0.0, 169.0, 1e4, :I),
    _marrero_mason_12("Ar", "Xe", 5.0, 1.563, 36.8, 59.9, 0.0, 169.0, 1e4, :I),
    _marrero_mason_12("Ar", "H2", 23.5, 1.519, 0.488, 39.8, 0.0, 242.0, 1e4, :II),
    _marrero_mason_12("Kr", "Xe", 2.933, 1.608, 128.0, 52.7, 0.0, 169.0, 1e4, :I),
    _marrero_mason_12("Kr", "H2", 18.2, 1.564, 1.69, 26.4, 0.0, 77.0, 1e4, :II),
    _marrero_mason_12("H2", "D2", 24.7, 1.5, 0.0636, 6.072, 38.1, 14.0, 1e4, :II),
    _marrero_mason_12("H2", "N2", 15.39, 1.548, 0.316, -2.8, 1067.0, 65.0, 1e4, :I),
    _marrero_mason_12("H2", "CO", 15.39, 1.548, 0.316, -2.8, 1067.0, 65.0, 1e4, :II),
    _marrero_mason_12("N2", "CO", 4.4, 1.576, 1.57, -36.2, 3825.0, 78.0, 1e4, :II),
    # Table 13, eq (4.3-2)
    _marrero_mason_13("He", "CH4", 3.13, 1.75, 0.0, 298.0, 1e4, :III),
    _marrero_mason_13("He", "O2", 4.37, 1.71, 0.0, 244.0, 1e4, :II),
    _marrero_mason_13("He", "air", 3.78, 1.729, 0.0, 244.0, 1e4, :II),
    _marrero_mason_13("He", "CO2", 3.31, 1.72, 0.0, 200.0, 530.0, :II),
    _marrero_mason_13("He", "SF6", 3.87, 1.627, 0.0, 290.0, 1e4, :III),
    _marrero_mason_13("Ne", "H2", 5.95, 1.731, 0.0, 90.0, 1e4, :II),
    _marrero_mason_13("Ne", "N2", 1.59, 1.743, 0.0, 293.0, 1e4, :III),
    _marrero_mason_13("Ne", "CO2", 1.07, 1.776, 0.0, 195.0, 625.0, :miscellaneous),
    _marrero_mason_13("Ar", "CH4", 0.784, 1.785, 0.0, 307.0, 1e4, :III),
    _marrero_mason_13("Ar", "N2", 0.904, 1.752, 0.0, 244.0, 1e4, :II),
    _marrero_mason_13("Ar", "CO", 0.904, 1.752, 0.0, 244.0, 1e4, :III),
    _marrero_mason_13("Ar", "O2", 0.977, 1.736, 0.0, 243.0, 1e4, :III),
    _marrero_mason_13("Ar", "air", 0.917, 1.749, 0.0, 244.0, 1e4, :III),
    _marrero_mason_13("Ar", "CO2", 1.74, 1.646, 89.1, 276.0, 1800.0, :III),
    _marrero_mason_13("Ar", "SF6", 1.48, 1.596, 145.4, 328.0, 1e4, :III),
    _marrero_mason_13("Kr", "N2", 0.653, 1.766, 0.0, 248.0, 1e4, :III),
    _marrero_mason_13("Kr", "CO", 0.653, 1.766, 0.0, 248.0, 1e4, :III),
    _marrero_mason_13("Xe", "H2", 3.68, 1.712, 16.9, 242.0, 1e4, :III),
    _marrero_mason_13("Xe", "N2", 0.47, 1.789, 0.0, 242.0, 1e4, :III),
    _marrero_mason_13("H2", "CH4", 3.13, 1.765, 0.0, 293.0, 1e4, :III),
    _marrero_mason_13("H2", "O2", 4.17, 1.732, 0.0, 252.0, 1e4, :III),
    _marrero_mason_13("H2", "air", 3.64, 1.75, 0.0, 252.0, 1e4, :II),
    _marrero_mason_13("H2", "CO2", 3.14, 1.75, 11.7, 200.0, 550.0, :II),
    _marrero_mason_13("H2", "SF6", 7.82, 1.57, 102.3, 298.0, 1e4, :III),
    _marrero_mason_13("CH4", "N2", 1.0, 1.75, 0.0, 298.0, 1e4, :III),
    _marrero_mason_13("CH4", "O2", 1.68, 1.695, 44.2, 294.0, 1e4, :III),
    _marrero_mason_13("CH4", "air", 1.03, 1.747, 0.0, 298.0, 1e4, :III),
    _marrero_mason_13("CH4", "SF6", 1.1, 1.657, 69.2, 298.0, 1e4, :III),
    _marrero_mason_13("N2", "O2", 1.13, 1.724, 0.0, 285.0, 1e4, :III),
    _marrero_mason_13("N2", "H2O", 0.187, 2.072, 0.0, 282.0, 373.0, :miscellaneous),
    _marrero_mason_13("N2", "CO2", 3.15, 1.57, 113.6, 288.0, 1800.0, :II),
    _marrero_mason_13("N2", "SF6", 1.66, 1.59, 119.4, 328.0, 1e4, :III),
    _marrero_mason_13("CO", "O2", 1.13, 1.724, 0.0, 285.0, 1e4, :III),
    _marrero_mason_13("CO", "air", 1.12, 1.73, 0.0, 285.0, 1e4, :III),
    _marrero_mason_13("CO", "CO2", 0.577, 1.803, 0.0, 282.0, 473.0, :III),
    _marrero_mason_13("CO", "SF6", 1.76, 1.584, 139.4, 297.0, 1e4, :III),
    _marrero_mason_13("O2", "H2O", 0.189, 2.072, 0.0, 282.0, 450.0, :miscellaneous),
    _marrero_mason_13("O2", "H2O", 2.78, 1.632, 0.0, 450.0, 1070.0, :miscellaneous),
    _marrero_mason_13("O2", "CO2", 1.56, 1.661, 61.3, 287.0, 1083.0, :III),
    _marrero_mason_13("O2", "SF6", 2.65, 1.522, 129.0, 297.0, 1e4, :III),
    _marrero_mason_13("air", "H2O", 0.187, 2.072, 0.0, 282.0, 450.0, :miscellaneous),
    _marrero_mason_13("air", "H2O", 2.75, 1.632, 0.0, 450.0, 1070.0, :miscellaneous),
    _marrero_mason_13("air", "CO2", 2.7, 1.59, 102.1, 280.0, 1800.0, :III),
    _marrero_mason_13("air", "SF6", 1.83, 1.576, 121.1, 328.0, 1e4, :III),
    _marrero_mason_13("H2O", "CO2", 9.24, 1.5, 307.9, 296.0, 1640.0, :miscellaneous),
    _marrero_mason_13("CO2", "N2O", 0.281, 1.866, 0.0, 195.0, 550.0, :III),
    _marrero_mason_13("CO2", "C3H8", 0.177, 1.896, 0.0, 298.0, 550.0, :miscellaneous),
    _marrero_mason_13("CO2", "SF6", 0.14, 1.886, 0.0, 328.0, 472.0, :III),
    _marrero_mason_13("H", "He", 14.2, 1.732, 0.0, 275.0, 1e4, :miscellaneous),
    _marrero_mason_13("H", "Ar", 1.45, 1.597, 0.0, 275.0, 1e4, :miscellaneous),
    _marrero_mason_13("H", "H2", 11.3, 1.728, 0.0, 190.0, 1e4, :miscellaneous),
    _marrero_mason_13("N", "N2", 1.32, 1.774, 0.0, 280.0, 1e4, :miscellaneous),
    _marrero_mason_13("O", "He", 4.68, 1.749, 0.0, 280.0, 1e4, :miscellaneous),
    _marrero_mason_13("O", "Ar", 0.751, 1.841, 0.0, 280.0, 1e4, :miscellaneous),
    _marrero_mason_13("O", "N2", 1.32, 1.774, 0.0, 280.0, 1e4, :miscellaneous),
    _marrero_mason_13("O", "O2", 1.32, 1.774, 0.0, 280.0, 1e4, :miscellaneous),
]

"""
    MARRERO_MASON_UNCERTAINTY

The estimated uncertainty limits of Marrero and Mason (1972), figure 4, in
percent, by reliability group at the tabulated temperatures in K. A missing
entry (`NaN`) is one the paper leaves blank. Group `:miscellaneous` pairs
carry per-system limits in the paper's Table 11 and are not tabulated here.
"""
const MARRERO_MASON_UNCERTAINTY = (
    temperature = (1.75, 65.0, 300.0, 500.0, 1e3, 1e4),
    I = (NaN, 2.0, 1.0, 2.0, 5.0, 10.0),
    II = (6.0, 4.0, 2.0, 3.0, 7.0, 15.0),
    III = (NaN, NaN, 3.0, 4.0, 10.0, 20.0),
)

"""
    marrero_mason_pairs(a, b)

Every [`MarreroMasonPair`](@ref) row of species `a` and `b` in either
order, in table order; empty when the paper does not correlate that pair.
Names are matched exactly.
"""
function marrero_mason_pairs(a::AbstractString, b::AbstractString)
    return [pair for pair in MARRERO_MASON_1972
            if pair.species == (a, b) || pair.species == (b, a)]
end
marrero_mason_pairs(a, b) = marrero_mason_pairs(String(a), String(b))

"""
    marrero_mason_pair(a, b; temperature=nothing)

The [`MarreroMasonPair`](@ref) of species `a` and `b` in either order, or
`nothing` when the paper does not correlate that pair. A pair with several
rows needs a `temperature` in K, and the row with the narrowest stated
range containing it is returned, so the paper's dedicated low-temperature
or interval correlation wins where it exists; `nothing` when no row
contains the temperature.
"""
function marrero_mason_pair(a::AbstractString, b::AbstractString; temperature=nothing)
    rows = marrero_mason_pairs(a, b)
    isempty(rows) && return nothing
    length(rows) == 1 && temperature === nothing && return only(rows)
    temperature === nothing && throw(ArgumentError(
        "the $a-$b pair has $(length(rows)) correlation ranges; " *
        "pass temperature to select one"))
    best = nothing
    for pair in rows
        pair.temperature_min <= temperature <= pair.temperature_max || continue
        if best === nothing || pair.temperature_max - pair.temperature_min <
                               best.temperature_max - best.temperature_min
            best = pair
        end
    end
    return best
end
marrero_mason_pair(a, b; temperature=nothing) =
    marrero_mason_pair(String(a), String(b); temperature=temperature)

# p D_12 in Pa m^2/s from the paper's equation (4.3-1) or (4.3-2).
function _marrero_mason_pD(pair::MarreroMasonPair, temperature)
    log_pD = log(pair.A) + pair.s * log(temperature) - pair.S / temperature
    if pair.phi0_over_k > 0
        log_pD -= 2 * log(log(pair.phi0_over_k / temperature)) +
                  pair.S_prime / temperature^2
    end
    return exp(log_pD)
end

"""
    marrero_mason_diffusivity(pair, temperature, pressure)

The binary diffusion coefficient of a [`MarreroMasonPair`](@ref) in m²/s at
a temperature in K and a pressure in Pa, from the paper's correlation with
its dilute-gas ``1/p`` scaling. The temperature must lie in the pair's
stated range.
"""
function marrero_mason_diffusivity(pair::MarreroMasonPair, temperature, pressure)
    isfinite(temperature) && pair.temperature_min <= temperature <= pair.temperature_max ||
        throw(ArgumentError("temperature $temperature K lies outside the " *
                            "$(pair.species[1])-$(pair.species[2]) correlation range " *
                            "$(pair.temperature_min) to $(pair.temperature_max) K"))
    isfinite(pressure) && pressure > 0 ||
        throw(ArgumentError("pressure must be finite and positive"))
    return _marrero_mason_pD(pair, temperature) / pressure
end

# Calculated hydrogen-isotopologue and helium pairs from B. Song, K. Kang,
# Z. Zhang, X. Wang and Z. Liu, "Ab Initio Values of the Gas Transport
# Properties of Hydrogen Isotopologues and Helium--Hydrogen Mixtures at Low
# Density", J. Chem. Eng. Data 61, 1910--1916 (2016),
# https://doi.org/10.1021/acs.jced.6b00076: classical kinetic theory on the
# spherically averaged ab initio potentials, tabulated in the supporting
# information at 298.15--2000 K and 101.3 kPa for mole fractions 0.25, 0.50
# and 0.75. The rows are degree-4 fits in log(T / 300 K) to the x1 = 0.50
# tables, produced by data/songwang_extract.jl from the publisher's PDF. A
# second Xpdf/layout extraction in data/songwang_verify.jl checks all 78
# mixture tables and 702 equimolar nodes independently of the PyMuPDF
# cell-order extraction. The tables themselves are not vendored.

"""
    SongWangPair

One calculated pair of Song et al. (2016): a degree-4 polynomial fit in
``\\log(T/300\\,\\mathrm{K})`` to the tabulated equimolar binary diffusion
coefficient, ``D_{12}(T) = D_{ref}\\exp(\\sum_m c_m z^m)``, with `D_ref` in
m²/s at 300 K and `pressure_ref` (101.3 kPa), scaled by ``1/p``.
`temperature_min` and `temperature_max` are the tabulated range,
`node_residual` the largest relative departure of the fit from the 27
tabulated nodes, and `composition_spread` the largest relative departure
of the 0.25 and 0.75 mole-fraction tables from the equimolar one. The
paper's expanded uncertainty is [`SONG_WANG_UNCERTAINTY`](@ref). These
are collision-model predictions, not measurements, and `"3He"`, `"4He"`
are the isotopes the paper names.
"""
struct SongWangPair
    species::NTuple{2,String}
    D_ref::Float64
    coefficients::NTuple{4,Float64}
    temperature_min::Float64
    temperature_max::Float64
    pressure_ref::Float64
    node_residual::Float64
    composition_spread::Float64
end

const _SONG_WANG_TEMPERATURE_REF = 300.0

function _song_wang(a, b, D_ref, coefficients, node_residual, composition_spread)
    return SongWangPair((a, b), D_ref, coefficients, 298.15, 2000.0, 101.3e3, node_residual,
                        composition_spread)
end

"""
    SONG_WANG_UNCERTAINTY

The combined expanded uncertainty (``k = 2``) of the Song et al. (2016)
binary diffusion coefficients, as a fraction.
"""
const SONG_WANG_UNCERTAINTY = 0.02

"""
    SONG_WANG_2016

The calculated pairs of Song et al. (2016) as [`SongWangPair`](@ref) rows:
every pair of 3He, 4He, H2, HD, HT, D2, DT and T2 except H2-D2, which the
supporting information omits. Look a pair up with
[`song_wang_pair`](@ref) and evaluate it with
[`song_wang_diffusivity`](@ref).
"""
const SONG_WANG_2016 = SongWangPair[
    # x1 = 0.50 fits in log(T / 300 K), 298.15 to 2000 K, 101.3 kPa
    _song_wang("3He", "D2", 0.00014008887,
               (1.7130375, 0.010490386, 0.0035178326, -0.00015437738), 4.0e-05, 2.8e-03),
    _song_wang("3He", "DT", 0.00013421357,
               (1.7126977, 0.010475285, 0.0034729388, -0.00013908759), 2.8e-05, 3.5e-03),
    _song_wang("3He", "H2", 0.00016729221,
               (1.7130924, 0.010507176, 0.0035577443, -0.00017082005), 2.9e-05, 3.0e-03),
    _song_wang("3He", "HD", 0.0001494651,
               (1.7133572, 0.010549591, 0.0035416306, -0.00016416223), 5.0e-05, 2.0e-03),
    _song_wang("3He", "HT", 0.00014012,
               (1.7131242, 0.010241884, 0.0037491473, -0.00021986845), 2.3e-05, 2.8e-03),
    _song_wang("3He", "T2", 0.00013017118,
               (1.7124206, 0.010337878, 0.0035448103, -0.00015479822), 2.4e-05, 4.3e-03),
    _song_wang("4He", "D2", 0.00012960312,
               (1.7132621, 0.010779459, 0.0033568845, -0.00011622243), 2.8e-05, 1.9e-03),
    _song_wang("4He", "DT", 0.00012317876,
               (1.7129464, 0.010766266, 0.0033634849, -0.00012227515), 3.2e-05, 2.6e-03),
    _song_wang("4He", "H2", 0.00015912004,
               (1.7127804, 0.010386553, 0.0035894588, -0.00017664781), 3.1e-05, 4.0e-03),
    _song_wang("4He", "HD", 0.00014002909,
               (1.7132488, 0.010523723, 0.0035338587, -0.00015835245), 3.8e-05, 2.7e-03),
    _song_wang("4He", "HT", 0.000129638,
               (1.7132928, 0.010599979, 0.0035650045, -0.0001816002), 2.2e-05, 2.0e-03),
    _song_wang("4He", "T2", 0.00011871338,
               (1.7129804, 0.010180911, 0.0037442292, -0.0002112221), 3.6e-05, 3.2e-03),
    _song_wang("D2", "DT", 0.00010030234,
               (1.7328324, -0.0062088511, 0.010074502, -0.0010250563), 3.0e-05, 2.2e-03),
    _song_wang("D2", "HD", 0.00011420654,
               (1.7328505, -0.0063086453, 0.010110475, -0.001029553), 5.7e-05, 2.4e-03),
    _song_wang("D2", "HT", 0.00010561876,
               (1.7326457, -0.0057209102, 0.00984622, -0.00097988392), 3.6e-05, 1.7e-03),
    _song_wang("D2", "T2", 9.662604e-05,
               (1.7330011, -0.006708558, 0.010346452, -0.0010852082), 4.5e-05, 2.7e-03),
    _song_wang("H2", "DT", 0.00012561268,
               (1.7332499, -0.0075113077, 0.010464119, -0.0010698201), 3.8e-05, 4.4e-03),
    _song_wang("H2", "HD", 0.00013656704,
               (1.7329609, -0.0066219618, 0.010254007, -0.0010559369), 5.2e-05, 2.7e-03),
    _song_wang("H2", "HT", 0.00012982505,
               (1.7330634, -0.0069563914, 0.010270337, -0.0010450076), 5.5e-05, 3.6e-03),
    _song_wang("H2", "T2", 0.00012275217,
               (1.7332491, -0.0076212651, 0.01038033, -0.0010367685), 3.5e-05, 5.0e-03),
    _song_wang("HD", "DT", 0.00010936928,
               (1.7328671, -0.0065007036, 0.010108906, -0.0010220675), 3.3e-05, 3.0e-03),
    _song_wang("HD", "HT", 0.00011423192,
               (1.7327911, -0.0061371729, 0.0099621431, -0.0009899628), 4.0e-05, 2.4e-03),
    _song_wang("HT", "DT", 0.0001003324,
               (1.7328277, -0.0061680763, 0.010004658, -0.00099702324), 5.2e-05, 2.3e-03),
    _song_wang("T2", "DT", 9.0559077e-05,
               (1.7328126, -0.0060903976, 0.0099821686, -0.00099864768), 5.4e-05, 2.1e-03),
    _song_wang("T2", "HD", 0.00010603285,
               (1.7329931, -0.006837803, 0.010164138, -0.0010134491), 3.4e-05, 3.6e-03),
    _song_wang("T2", "HT", 9.6657691e-05,
               (1.7328895, -0.0063155892, 0.0099537037, -0.0009724231), 6.7e-05, 2.7e-03),
]

"""
    song_wang_pair(a, b)

The [`SongWangPair`](@ref) of species `a` and `b` in either order, or
`nothing` when the paper does not tabulate that pair. Names are matched
exactly; helium is `"3He"` or `"4He"`.
"""
function song_wang_pair(a::AbstractString, b::AbstractString)
    for pair in SONG_WANG_2016
        (pair.species == (a, b) || pair.species == (b, a)) && return pair
    end
    return nothing
end
song_wang_pair(a, b) = song_wang_pair(String(a), String(b))

"""
    song_wang_diffusivity(pair, temperature, pressure)

The binary diffusion coefficient of a [`SongWangPair`](@ref) in m²/s at a
temperature in K inside the tabulated range and a pressure in Pa.
"""
function song_wang_diffusivity(pair::SongWangPair, temperature, pressure)
    isfinite(temperature) && pair.temperature_min <= temperature <= pair.temperature_max ||
        throw(ArgumentError("temperature $temperature K lies outside the " *
                            "$(pair.species[1])-$(pair.species[2]) tabulated range " *
                            "$(pair.temperature_min) to $(pair.temperature_max) K"))
    isfinite(pressure) && pressure > 0 ||
        throw(ArgumentError("pressure must be finite and positive"))
    z = log(temperature / _SONG_WANG_TEMPERATURE_REF)
    exponent = 0.0
    for m in 4:-1:1
        exponent = (exponent + pair.coefficients[m]) * z
    end
    return pair.D_ref * exp(exponent) * pair.pressure_ref / pressure
end

# Measured room-temperature hydrogen-isotopologue pairs from K. P. Müller and
# A. Klemm, "Diffusion in binären Gemischen von H2, HD und D2 bei 24 °C",
# Z. Naturforsch. 25a, 243--246 (1970), https://doi.org/10.1515/zna-1970-0216,
# Tab. 1: Loschmidt-cell measurements analysed catarometrically at 24 °C and
# 760 Torr. The H2-HD, H2-D2 and HD-D2 rows are that paper's own values with
# a stated mean relative error of 1%; the six tritiated rows are the older
# gas-counter measurements of Reichenbacher, Müller and Klemm, Z. Naturforsch.
# 20a, 1529 (1965), at 2%. The table was transcribed twice from the page
# image and diffed. The paper's adjusted values (its equation 1, a
# least-squares mass expansion holding H2-D2 fixed) are kept beside the
# measurements and are not measurements.

"""
    MuellerKlemmPair

One measured pair of Müller and Klemm (1970), Tab. 1: `D` and its printed
mean error in m²/s at 24 °C ([`MUELLER_KLEMM_TEMPERATURE`](@ref)) and
1 atm ([`MUELLER_KLEMM_PRESSURE`](@ref)), the paper's adjusted value
`D_adjusted` from its mass expansion, and `newly_measured`, `true` for the
paper's own catarometric H2-HD, H2-D2 and HD-D2 values and `false` for
the tritiated pairs it carries over from Reichenbacher et al. (1965).
"""
struct MuellerKlemmPair
    species::NTuple{2,String}
    D::Float64
    uncertainty::Float64
    D_adjusted::Float64
    newly_measured::Bool
end

"""Temperature of the Müller--Klemm (1970) measurements, 24 °C in K."""
const MUELLER_KLEMM_TEMPERATURE = 297.15
"""Pressure of the Müller--Klemm (1970) measurements, 760 Torr in Pa."""
const MUELLER_KLEMM_PRESSURE = _ATMOSPHERE

# Tab. 1 rows as printed, in cm^2/s: (D gemessen, its ±, D ausgeglichen).
function _mueller_klemm(a, b, D, uncertainty, D_adjusted, newly_measured)
    return MuellerKlemmPair((a, b), D * 1e-4, uncertainty * 1e-4, D_adjusted * 1e-4,
                            newly_measured)
end

"""
    MUELLER_KLEMM_1970

Tab. 1 of Müller and Klemm (1970) as [`MuellerKlemmPair`](@ref) rows, in
printed order. Look a pair up with [`mueller_klemm_pair`](@ref).
"""
const MUELLER_KLEMM_1970 = MuellerKlemmPair[
    _mueller_klemm("H2", "HD", 1.349, 0.013, 1.342, true),
    _mueller_klemm("H2", "HT", 1.274, 0.025, 1.274, false),
    _mueller_klemm("H2", "D2", 1.268, 0.013, 1.268, true),
    _mueller_klemm("H2", "DT", 1.212, 0.024, 1.223, false),
    _mueller_klemm("H2", "T2", 1.207, 0.030, 1.190, false),
    _mueller_klemm("HD", "D2", 1.126, 0.011, 1.127, true),
    _mueller_klemm("D2", "HT", 1.044, 0.021, 1.047, false),
    _mueller_klemm("D2", "DT", 0.989, 0.020, 0.989, false),
    _mueller_klemm("D2", "T2", 0.956, 0.019, 0.951, false),
]

"""
    mueller_klemm_pair(a, b)

The [`MuellerKlemmPair`](@ref) of species `a` and `b` in either order, or
`nothing` when Tab. 1 has no such row.
"""
function mueller_klemm_pair(a::AbstractString, b::AbstractString)
    for pair in MUELLER_KLEMM_1970
        (pair.species == (a, b) || pair.species == (b, a)) && return pair
    end
    return nothing
end
mueller_klemm_pair(a, b) = mueller_klemm_pair(String(a), String(b))

const _NEUTRAL_SOURCES = (:marrero_mason, :song_wang, :mueller_klemm)

# The first source of `source` that carries the pair at `temperature_ref`, with
# the row itself.
function _neutral_source(a, b, source, temperature_ref)
    sources = source isa Symbol ? (source,) : Tuple(source)
    for s in sources
        s in _NEUTRAL_SOURCES || throw(ArgumentError(
            "unknown binary diffusion source $s; use one of $(_NEUTRAL_SOURCES)"))
        row = if s === :marrero_mason
            marrero_mason_pair(a, b; temperature=temperature_ref)
        elseif s === :song_wang
            candidate = song_wang_pair(a, b)
            candidate !== nothing &&
                candidate.temperature_min <= temperature_ref <= candidate.temperature_max ?
                candidate : nothing
        else
            candidate = mueller_klemm_pair(a, b)
            candidate !== nothing && temperature_ref == MUELLER_KLEMM_TEMPERATURE ?
                candidate : nothing
        end
        row === nothing || return (s, row)
    end
    return nothing
end

"""
    neutral_binary_sources(species; source=:marrero_mason, temperature_ref=300)

Which source [`neutral_binary_diffusion`](@ref) takes each pair of
`species` from, as a `Dict` from the pair to `:marrero_mason`,
`:song_wang` or `:mueller_klemm`; a pair no listed source carries maps to
`nothing`. A source is eligible only when `temperature_ref` lies in that
source's stated range. `source` is one of those symbols or a tuple of them in
order of preference.
"""
function neutral_binary_sources(species; source=:marrero_mason, temperature_ref=300.0)
    names = [String(name) for name in species]
    out = Dict{NTuple{2,String},Union{Symbol,Nothing}}()
    for i in eachindex(names), j in i+1:length(names)
        found = _neutral_source(names[i], names[j], source, temperature_ref)
        out[(names[i], names[j])] = found === nothing ? nothing : found[1]
    end
    return out
end

"""
    neutral_binary_diffusion(species; source=:marrero_mason, degree=10,
                             temperature_ref=300, pressure_ref=101325,
                             temperature_min=nothing, temperature_max=nothing)

A [`BinaryDiffusionPolynomial`](@ref) for every pair of `species`, each pair
taken from the first entry of `source` that carries it at `temperature_ref`:
`:marrero_mason`, the evaluated correlations of
[`MARRERO_MASON_1972`](@ref); `:song_wang`, the calculated fits of
[`SONG_WANG_2016`](@ref); or `:mueller_klemm`, the room-temperature
measurements of [`MUELLER_KLEMM_1970`](@ref). The default is Marrero and
Mason alone; a hydrogen-isotopologue mixture with HD or tritium needs
`source=(:marrero_mason, :song_wang)`, and
[`neutral_binary_sources`](@ref) reports the choice made for each pair.
A pair no listed source carries is an error, not an estimate: no isotope
scaling or combination rule fills it in.

A Marrero and Mason pair is fitted in ``\\log(T/T_{ref})`` by least squares
on Chebyshev nodes over its stated range, or over the range's intersection
with `temperature_min`/`temperature_max`; the default degree holds every
row within a tenth of a percent of its source. A Song and Wang pair's
degree-4 polynomial is re-centred on `temperature_ref` exactly and padded
to `degree`, which must be at least 4. A Müller and Klemm pair is valid at
its measurement temperature only, so `temperature_ref` must equal
[`MUELLER_KLEMM_TEMPERATURE`](@ref) and the pair range collapses to that
point. [`neutral_binary_diffusion_residual`](@ref) measures each pair
against its source, and the sources' own uncertainties are in
[`MARRERO_MASON_UNCERTAINTY`](@ref), [`SONG_WANG_UNCERTAINTY`](@ref) and
the `uncertainty` field of a Müller and Klemm row.
"""
function neutral_binary_diffusion(species; source=:marrero_mason, degree::Integer=10,
                                  temperature_ref=300.0, pressure_ref=_ATMOSPHERE,
                                  temperature_min=nothing, temperature_max=nothing)
    names = [String(name) for name in species]
    N = length(names)
    N >= 2 || throw(ArgumentError("neutral_binary_diffusion needs at least two species"))
    degree >= 1 || throw(ArgumentError("polynomial degree must be at least 1"))
    D_ref = zeros(N, N)
    coefficients = zeros(N, N, degree)
    Tmin = zeros(N, N)
    Tmax = zeros(N, N)
    for i in 1:N, j in i+1:N
        found = _neutral_source(names[i], names[j], source, temperature_ref)
        found === nothing && throw(ArgumentError(
            "no listed source carries a $(names[i])-$(names[j]) pair at " *
            "$temperature_ref K; supply a sourced fit for it"))
        which, pair = found
        lo, hi, c, D = _neutral_pair_fit(which, pair, names[i], names[j], degree,
                                         temperature_ref, pressure_ref, temperature_min,
                                         temperature_max)
        D_ref[i, j] = D_ref[j, i] = D
        coefficients[i, j, :] .= c
        coefficients[j, i, :] .= c
        Tmin[i, j] = Tmin[j, i] = lo
        Tmax[i, j] = Tmax[j, i] = hi
    end
    return BinaryDiffusionPolynomial(names, D_ref, coefficients, Tmin, Tmax;
                                     temperature_ref=temperature_ref,
                                     pressure_ref=pressure_ref)
end

function _neutral_range(pair, a, b, temperature_ref, temperature_min, temperature_max)
    lo = temperature_min === nothing ? pair.temperature_min :
         max(pair.temperature_min, float(temperature_min))
    hi = temperature_max === nothing ? pair.temperature_max :
         min(pair.temperature_max, float(temperature_max))
    lo < hi || throw(ArgumentError(
        "the requested range leaves no $a-$b interval inside " *
        "$(pair.temperature_min) to $(pair.temperature_max) K"))
    lo <= temperature_ref <= hi || throw(ArgumentError(
        "temperature_ref must lie in the $a-$b range $lo to $hi K"))
    return lo, hi
end

function _neutral_pair_fit(::Val{:marrero_mason}, pair::MarreroMasonPair, a, b, degree,
                           temperature_ref, pressure_ref, temperature_min, temperature_max)
    lo, hi = _neutral_range(pair, a, b, temperature_ref, temperature_min, temperature_max)
    c = _fit_log_polynomial(pair, lo, hi, temperature_ref, degree)
    return lo, hi, c, marrero_mason_diffusivity(pair, temperature_ref, pressure_ref)
end

function _neutral_pair_fit(::Val{:song_wang}, pair::SongWangPair, a, b, degree,
                           temperature_ref, pressure_ref, temperature_min, temperature_max)
    degree >= 4 || throw(ArgumentError(
        "a Song and Wang pair needs polynomial degree 4 or more, got $degree"))
    lo, hi = _neutral_range(pair, a, b, temperature_ref, temperature_min, temperature_max)
    c = zeros(degree)
    c[1:4] .= _recentre(pair.coefficients, log(temperature_ref / _SONG_WANG_TEMPERATURE_REF))
    return lo, hi, c, song_wang_diffusivity(pair, temperature_ref, pressure_ref)
end

function _neutral_pair_fit(::Val{:mueller_klemm}, pair::MuellerKlemmPair, a, b, degree,
                           temperature_ref, pressure_ref, temperature_min, temperature_max)
    temperature_ref == MUELLER_KLEMM_TEMPERATURE || throw(ArgumentError(
        "a Müller and Klemm pair is measured at $MUELLER_KLEMM_TEMPERATURE K only; " *
        "set temperature_ref to it"))
    for bound in (temperature_min, temperature_max)
        bound === nothing || bound == MUELLER_KLEMM_TEMPERATURE || throw(ArgumentError(
            "a Müller and Klemm pair admits no range other than its measurement temperature"))
    end
    D = pair.D * MUELLER_KLEMM_PRESSURE / pressure_ref
    return MUELLER_KLEMM_TEMPERATURE, MUELLER_KLEMM_TEMPERATURE, zeros(degree), D
end

_neutral_pair_fit(which::Symbol, args...) = _neutral_pair_fit(Val(which), args...)

# The coefficients of sum_m c_m (u + delta)^m, minus its constant term, as a
# polynomial in u: a polynomial in z = log(T / T_old) re-expressed in
# u = log(T / T_new) with delta = log(T_new / T_old). The constant term is
# absorbed into the reference value.
function _recentre(c::NTuple{M,Float64}, delta) where {M}
    out = zeros(M)
    for m in 1:M, k in 1:m
        out[k] += c[m] * binomial(m, k) * delta^(m - k)
    end
    return out
end

# Least-squares coefficients c_m of log(pD(T)/pD(Tref)) = sum_m c_m z^m on
# Chebyshev nodes in z = log(T/Tref); the pressure factor cancels. The
# solve runs in u = z / scale with |u| <= 1, since a range of four decades
# puts |z| near 9 and the unscaled Vandermonde loses ten digits.
function _fit_log_polynomial(pair::MarreroMasonPair, lo, hi, temperature_ref, degree;
                             nodes::Integer=64 * degree)
    zmin, zmax = log(lo / temperature_ref), log(hi / temperature_ref)
    z = [(zmin + zmax) / 2 + (zmax - zmin) / 2 * cospi((k + 0.5) / nodes) for k in 0:nodes-1]
    log_ratio = [log(_marrero_mason_pD(pair, temperature_ref * exp(zk)) /
                     _marrero_mason_pD(pair, temperature_ref)) for zk in z]
    scale = max(abs(zmin), abs(zmax))
    vandermonde = [(zk / scale)^m for zk in z, m in 1:degree]
    scaled = vandermonde \ log_ratio
    return [scaled[m] / scale^m for m in 1:degree]
end

"""
    neutral_binary_diffusion_residual(model, i, j; source=:marrero_mason,
                                      samples=2000)

The largest relative departure of pair `(i, j)` of a
[`BinaryDiffusionPolynomial`](@ref) built by
[`neutral_binary_diffusion`](@ref) from its source, resolved with the
same `source` preference: against the Marrero and Mason correlation on
`samples` log-spaced temperatures over the pair's range, or against the
Song and Wang fit likewise with that fit's own node residual added, or
against the Müller and Klemm measurement at its temperature.
"""
function neutral_binary_diffusion_residual(model::BinaryDiffusionPolynomial, i::Integer,
                                           j::Integer; source=:marrero_mason,
                                           samples::Integer=2000)
    names = species_names(model)
    found = _neutral_source(names[i], names[j], source, model.temperature_ref)
    found === nothing && throw(ArgumentError(
        "no listed source carries a $(names[i])-$(names[j]) pair at " *
        "$(model.temperature_ref) K"))
    which, pair = found
    lo, hi = model.temperature_min[i][j], model.temperature_max[i][j]
    worst = which === :song_wang ? pair.node_residual : 0.0
    for temperature in (lo == hi ? (lo,) : _log_spaced(lo, hi, samples))
        fitted = binary_diffusivity(model, temperature, model.pressure_ref, i, j)
        reference = which === :marrero_mason ?
            marrero_mason_diffusivity(pair, temperature, model.pressure_ref) :
            which === :song_wang ?
            song_wang_diffusivity(pair, temperature, model.pressure_ref) :
            pair.D * MUELLER_KLEMM_PRESSURE / model.pressure_ref
        worst = max(worst, abs(fitted / reference - 1))
    end
    return worst
end

# Log-spaced samples whose end points are exactly the range bounds, since a
# round trip through exp and log lands a hair outside an inclusive range.
function _log_spaced(lo, hi, samples)
    temperatures = exp.(range(log(lo), log(hi), length=samples))
    temperatures[1] = lo
    temperatures[end] = hi
    return temperatures
end

"""
    temperature_domain(model)

The temperature interval, in K, on which every pair of a
[`BinaryDiffusionPolynomial`](@ref) is valid: the intersection of the pair
ranges, as `(temperature_min, temperature_max)`. A mixture evaluation must
stay inside it, since each pair is checked against its own range.
"""
function temperature_domain(model::BinaryDiffusionPolynomial{T,N}) where {T,N}
    lo, hi = typemin(T), typemax(T)
    for i in 1:N, j in i+1:N
        lo = max(lo, model.temperature_min[i][j])
        hi = min(hi, model.temperature_max[i][j])
    end
    return (lo, hi)
end
