# Gates on the vendored Marrero and Mason (1972) binary-diffusion
# correlations and the log-polynomial fits built from them: transcription
# against independent literals, units, the exact 1/p scaling, symmetry,
# positivity, fit residuals over every source range, source-range behavior,
# and agreement with the paper's tabulated curve-fit nodes within the
# scatter its deviation plots show. Included by runtests.jl and runnable
# directly.

using CompactLES
using Test

# Marrero and Mason, Table 12, H2-D2 row as printed: 10^3 A = 24.7,
# s = 1.500, 10^-8 phi0/k = 0.0636, S = 6.072, S' = 38.10, 14 to 10^4 K,
# group II. p D in atm cm^2/s, written out here independently of the package.
function h2_d2_source_pD(T)
    return 24.7e-3 * T^1.5 / (log(0.0636e8 / T)^2 * exp(6.072 / T) * exp(38.10 / T^2))
end

# Table 13, N2-O2 row as printed: 10^5 A = 1.13, s = 1.724, S blank,
# 285 to 10^4 K, group III; eq (4.3-2) in atm cm^2/s.
n2_o2_source_pD(T) = 1.13e-5 * T^1.724

const ATM = 101325.0

@testset "Marrero--Mason transcription and units" begin
    pair = marrero_mason_pair("H2", "D2")
    @test pair !== nothing
    @test marrero_mason_pair("D2", "H2") === pair
    @test marrero_mason_pair(:H2, :D2) === pair
    @test marrero_mason_pair("H2", "HD") === nothing
    @test marrero_mason_pairs("H2", "HD") == MarreroMasonPair[]
    @test pair.species == ("H2", "D2")
    # A is stored in Pa m^2 s^-1 K^-s: the printed atm cm^2 value times
    # 101325 Pa/atm times 1e-4 m^2/cm^2.
    @test pair.A ≈ 24.7e-3 * 10.1325 rtol=2e-16
    @test pair.s == 1.5
    @test pair.phi0_over_k == 6.36e6
    @test pair.S == 6.072
    @test pair.S_prime == 38.10
    @test (pair.temperature_min, pair.temperature_max) == (14.0, 1e4)
    @test pair.group === :II

    # SI: atm cm^2/s at 1 atm is 1e-4 m^2/s at 101325 Pa.
    for T in (14.0, 77.0, 300.0, 1000.0, 1e4)
        @test marrero_mason_diffusivity(pair, T, ATM) ≈ h2_d2_source_pD(T) * 1e-4 rtol=4e-15
    end
    @test marrero_mason_diffusivity(pair, 300.0, 2ATM) ==
          marrero_mason_diffusivity(pair, 300.0, ATM) / 2
    @test_throws ArgumentError marrero_mason_diffusivity(pair, prevfloat(14.0), ATM)
    @test_throws ArgumentError marrero_mason_diffusivity(pair, nextfloat(1e4), ATM)
    @test_throws ArgumentError marrero_mason_diffusivity(pair, 300.0, 0.0)

    n2_o2 = marrero_mason_pair("O2", "N2")
    @test n2_o2.species == ("N2", "O2") && n2_o2.group === :III
    @test n2_o2.phi0_over_k == 0 && n2_o2.S == 0 && n2_o2.S_prime == 0
    for T in (285.0, 300.0, 2000.0, 1e4)
        @test marrero_mason_diffusivity(n2_o2, T, ATM) ≈ n2_o2_source_pD(T) * 1e-4 rtol=4e-15
    end
    @test marrero_mason_diffusivity(n2_o2, 300.0, 3.0e6) ≈
          n2_o2_source_pD(300.0) * 1e-4 * ATM / 3.0e6 rtol=4e-15

    # A pair with two correlation ranges is selected by temperature; the
    # paper's rows join continuously at the shared bound.
    @test length(marrero_mason_pairs("H2O", "O2")) == 2
    @test_throws ArgumentError marrero_mason_pair("O2", "H2O")
    low = marrero_mason_pair("O2", "H2O"; temperature=300.0)
    high = marrero_mason_pair("O2", "H2O"; temperature=600.0)
    @test (low.temperature_min, low.temperature_max) == (282.0, 450.0)
    @test (high.temperature_min, high.temperature_max) == (450.0, 1070.0)
    @test marrero_mason_pair("O2", "H2O"; temperature=2000.0) === nothing
    @test marrero_mason_diffusivity(low, 450.0, ATM) ≈
          marrero_mason_diffusivity(high, 450.0, ATM) rtol=5e-3
    helium = marrero_mason_pair("3He", "4He"; temperature=36.0)
    @test (helium.temperature_min, helium.temperature_max) == (14.4, 90.0)
    @test marrero_mason_pair("3He", "4He"; temperature=300.0).temperature_max == 1e4

    uncertainty = MARRERO_MASON_UNCERTAINTY
    k = findfirst(==(300.0), uncertainty.temperature)
    @test uncertainty.II[k] == 2.0
    @test isnan(uncertainty.I[1]) && isnan(uncertainty.III[2])
end

# Marrero and Mason, Table 20, the H2-D2 block: temperature in K and
# log10 of p D_12 (x = 1/2) in atm cm^2/s, every weight 1, in printed order
# (the 90.0 K row is printed out of order). Notes: q and r are calculated
# from HD viscosity (Becker and Misenta 1955; Coremans et al. 1958), s from
# H2 viscosity (Mason and Rice 1954), t from the molecular-beam potential
# (Amdur et al.). The paper fitted its equation to these with s fixed at
# 1.500, so they are the fit's source nodes, not independent measurements.
const H2_D2_TABLE_20 = [
    (14.12, -2.3675), (15.47, -2.2832), (17.04, -2.1945), (18.70, -2.1051),
    (20.32, -2.0329), (90.0, -0.7721), (26.09, -1.8097), (32.57, -1.6091),
    (41.35, -1.4117), (48.06, -1.2832), (60.30, -1.1002), (70.32, -0.9851),
    (200.0, -0.1925), (250.0, -0.0292), (293.0, 0.0864), (400.0, 0.3181),
    (500.0, 0.4757), (763.0, 0.7882), (986.0, 0.9741), (3313.0, 1.9047),
    (5000.0, 2.2305), (10000.0, 2.7796),
]

@testset "Marrero--Mason H2-D2 source nodes" begin
    pair = marrero_mason_pair("H2", "D2")
    model = neutral_binary_diffusion(("H2", "D2"))
    for (T, log10_pD) in H2_D2_TABLE_20
        node = 10.0^log10_pD * 1e-4
        source = marrero_mason_diffusivity(pair, T, ATM)
        fitted = binary_diffusivity(model, T, ATM, 1, 2)
        # The paper's own equation scatters about its nodes by the amounts
        # its deviation plots show, under 7% here; the polynomial adds
        # nothing visible to that on top of the equation.
        @test abs(source / node - 1) < 0.07
        @test abs(fitted / source - 1) < 1e-4
    end
end

@testset "Marrero--Mason table integrity" begin
    @test length(MARRERO_MASON_1972) == 77
    seen = Dict{NTuple{2,String},Int}()
    for pair in MARRERO_MASON_1972
        a, b = pair.species
        @test a != b
        @test !haskey(seen, (b, a))
        seen[(a, b)] = get(seen, (a, b), 0) + 1
        @test pair.A > 0 && 1.4 < pair.s < 2.2
        @test pair.phi0_over_k >= 0 && pair.S_prime >= 0
        @test 0 < pair.temperature_min < pair.temperature_max <= 1e4
        @test pair.group in (:I, :II, :III, :miscellaneous)
        # Positive, finite and increasing in temperature over the stated range.
        D = [marrero_mason_diffusivity(pair, T, ATM)
             for T in CompactLES._log_spaced(pair.temperature_min, pair.temperature_max, 200)]
        @test all(isfinite, D) && D[1] > 0 && issorted(D; lt=(<=))
    end
    # 74 distinct systems, as the paper counts them.
    @test length(seen) == 74
    @test Set(k for (k, n) in seen if n == 2) ==
          Set([("3He", "4He"), ("O2", "H2O"), ("air", "H2O")])
end

@testset "neutral binary-diffusion polynomial fits" begin
    model = neutral_binary_diffusion(("H2", "D2"))
    @test isbitstype(typeof(model))
    @test species_names(model) == ["H2", "D2"]
    @test temperature_domain(model) == (14.0, 1e4)
    pair = marrero_mason_pair("H2", "D2")
    @test model.D_ref[1][2] == marrero_mason_diffusivity(pair, 300.0, ATM)
    @test @inferred(binary_diffusivity(model, 300.0, ATM, 1, 2)) == model.D_ref[1][2]
    @test binary_diffusivity(model, 500.0, ATM, 1, 2) ==
          binary_diffusivity(model, 500.0, ATM, 2, 1)
    @test binary_diffusivity(model, 500.0, 2ATM, 1, 2) ≈
          binary_diffusivity(model, 500.0, ATM, 1, 2) / 2 rtol=4e-16
    @test_throws ArgumentError binary_diffusivity(model, prevfloat(14.0), ATM, 1, 2)
    @test_throws ArgumentError binary_diffusivity(model, nextfloat(1e4), ATM, 1, 2)

    # The default degree over the full 14 to 10^4 K range stays two orders
    # below the source's smallest uncertainty limit; a practical range at
    # the same degree is far tighter, and a lower degree is coarser.
    @test neutral_binary_diffusion_residual(model, 1, 2) < 1e-4
    narrow = neutral_binary_diffusion(("H2", "D2"); temperature_min=200.0,
                                      temperature_max=2000.0)
    @test temperature_domain(narrow) == (200.0, 2000.0)
    @test neutral_binary_diffusion_residual(narrow, 1, 2) < 1e-9
    coarse = neutral_binary_diffusion(("H2", "D2"); degree=4)
    @test 1e-3 < neutral_binary_diffusion_residual(coarse, 1, 2) < 5e-2
    @test_throws ArgumentError binary_diffusivity(narrow, 199.0, ATM, 1, 2)

    # A mixture takes the intersection of its pair ranges as its domain, and
    # a species order permutation moves the pair values with the names.
    air = neutral_binary_diffusion(("N2", "O2", "CO2"))
    @test temperature_domain(air) == (288.0, 1083.0)
    @test binary_diffusivity(air, 1000.0, ATM, 1, 2) ≈
          n2_o2_source_pD(1000.0) * 1e-4 rtol=5e-4
    swapped = neutral_binary_diffusion(("CO2", "O2", "N2"))
    @test binary_diffusivity(swapped, 1000.0, ATM, 3, 2) ==
          binary_diffusivity(air, 1000.0, ATM, 1, 2)
    @test binary_diffusivity(swapped, 1000.0, ATM, 1, 3) ==
          binary_diffusivity(air, 1000.0, ATM, 1, 3)
    @test binary_diffusivity(air, 2000.0, ATM, 1, 2) > 0
    @test_throws ArgumentError binary_diffusivity(air, 2000.0, ATM, 2, 3)

    # A two-range pair fits the row selected at the reference temperature.
    steam = neutral_binary_diffusion(("O2", "H2O"); temperature_ref=600.0)
    @test temperature_domain(steam) == (450.0, 1070.0)
    @test neutral_binary_diffusion_residual(steam, 1, 2) < 1e-4

    # A pair the paper does not correlate is refused, never estimated.
    @test_throws ArgumentError neutral_binary_diffusion(("H2", "HD"))
    @test_throws ArgumentError neutral_binary_diffusion(("H2", "HD", "D2"))
    @test_throws ArgumentError neutral_binary_diffusion(("H2",))
    @test_throws ArgumentError neutral_binary_diffusion(("H2", "D2"); temperature_ref=5.0)
    @test_throws ArgumentError neutral_binary_diffusion(("H2", "D2"); temperature_min=2e4)
    @test_throws ArgumentError neutral_binary_diffusion(("H2", "D2"); degree=0)

    # Every vendored row fits its own full range to a tenth of a percent, an
    # order below the smallest source uncertainty limit, with the reference
    # at the geometric midpoint so a two-range pair's rows are each selected
    # in turn.
    for pair in MARRERO_MASON_1972
        lo, hi = pair.temperature_min, pair.temperature_max
        row = neutral_binary_diffusion(pair.species; temperature_ref=sqrt(lo * hi),
                                       temperature_min=lo, temperature_max=hi)
        @test temperature_domain(row) == (lo, hi)
        @test neutral_binary_diffusion_residual(row, 1, 2) < 1e-3
    end
end

# Song et al. (2016), supporting information, D12 in cm^2/s at 101.3 kPa
# and x1 = 0.50 on four of the 27 tabulated temperatures per pair, read
# from the publisher's text layer; the fits in SONG_WANG_2016 were made
# from all 27.
const SONG_WANG_CHECK_NODES = Dict(
    ("3He", "D2") => ((298.15, 1.3861), (500.0, 3.3715), (1000.0, 11.252), (2000.0, 38.349)),
    ("3He", "DT") => ((298.15, 1.328), (500.0, 3.2295), (1000.0, 10.775), (2000.0, 36.711)),
    ("3He", "H2") => ((298.15, 1.6553), (500.0, 4.0264), (1000.0, 13.439), (2000.0, 45.806)),
    ("3He", "HD") => ((298.15, 1.4789), (500.0, 3.5978), (1000.0, 12.012), (2000.0, 40.951)),
    ("3He", "HT") => ((298.15, 1.3864), (500.0, 3.3723), (1000.0, 11.255), (2000.0, 38.358)),
    ("3He", "T2") => ((298.15, 1.288), (500.0, 3.1317), (1000.0, 10.446), (2000.0, 35.579)),
    ("4He", "D2") => ((298.15, 1.2824), (500.0, 3.1197), (1000.0, 10.415), (2000.0, 35.509)),
    ("4He", "DT") => ((298.15, 1.2188), (500.0, 2.9646), (1000.0, 9.895), (2000.0, 33.726)),
    ("4He", "H2") => ((298.15, 1.5744), (500.0, 3.829), (1000.0, 12.776), (2000.0, 43.53)),
    ("4He", "HD") => ((298.15, 1.3855), (500.0, 3.3705), (1000.0, 11.251), (2000.0, 38.355)),
    ("4He", "HT") => ((298.15, 1.2827), (500.0, 3.1205), (1000.0, 10.418), (2000.0, 35.518)),
    ("4He", "T2") => ((298.15, 1.1746), (500.0, 2.8568), (1000.0, 9.5332), (2000.0, 32.484)),
    ("D2", "DT") => ((298.15, 0.99231), (500.0, 2.4299), (1000.0, 8.1315), (2000.0, 27.76)),
    ("D2", "HD") => ((298.15, 1.1299), (500.0, 2.7667), (1000.0, 9.2581), (2000.0, 31.604)),
    ("D2", "HT") => ((298.15, 1.0449), (500.0, 2.5587), (1000.0, 8.564), (2000.0, 29.244)),
    ("D2", "T2") => ((298.15, 0.95594), (500.0, 2.3408), (1000.0, 7.8321), (2000.0, 26.732)),
    ("H2", "DT") => ((298.15, 1.2427), (500.0, 3.0428), (1000.0, 10.175), (2000.0, 34.702)),
    ("H2", "HD") => ((298.15, 1.3511), (500.0, 3.3083), (1000.0, 11.069), (2000.0, 37.781)),
    ("H2", "HT") => ((298.15, 1.2844), (500.0, 3.1449), (1000.0, 10.52), (2000.0, 35.889)),
    ("H2", "T2") => ((298.15, 1.2144), (500.0, 2.9734), (1000.0, 9.9413), (2000.0, 33.893)),
    ("HD", "DT") => ((298.15, 1.082), (500.0, 2.6494), (1000.0, 8.8637), (2000.0, 30.248)),
    ("HD", "HT") => ((298.15, 1.1301), (500.0, 2.7673), (1000.0, 9.2602), (2000.0, 31.611)),
    ("HT", "DT") => ((298.15, 0.99261), (500.0, 2.4306), (1000.0, 8.1339), (2000.0, 27.769)),
    ("T2", "DT") => ((298.15, 0.89592), (500.0, 2.1939), (1000.0, 7.342), (2000.0, 25.066)),
    ("T2", "HD") => ((298.15, 1.049), (500.0, 2.5685), (1000.0, 8.5915), (2000.0, 29.311)),
    ("T2", "HT") => ((298.15, 0.95626), (500.0, 2.3415), (1000.0, 7.8346), (2000.0, 26.74)),
)

@testset "Song--Wang fits against their nodes" begin
    @test length(SONG_WANG_2016) == 26
    @test length(SONG_WANG_CHECK_NODES) == 26
    @test song_wang_pair("H2", "D2") === nothing
    @test song_wang_pair("He", "H2") === nothing
    for ((a, b), nodes) in SONG_WANG_CHECK_NODES
        pair = song_wang_pair(b, a)
        @test pair !== nothing && pair.species == (a, b)
        @test pair.node_residual < 1e-4 && pair.composition_spread < 6e-3
        @test (pair.temperature_min, pair.temperature_max) == (298.15, 2000.0)
        # The table's five significant digits bound how closely a smooth fit
        # can follow it; the check nodes sit inside the recorded residual,
        # which is itself rounded to two digits.
        for (T, D) in nodes
            departure = abs(song_wang_diffusivity(pair, T, 101.3e3) / (D * 1e-4) - 1)
            @test departure <= 1.1 * pair.node_residual
        end
        @test song_wang_diffusivity(pair, 500.0, 2 * 101.3e3) ==
              song_wang_diffusivity(pair, 500.0, 101.3e3) / 2
        @test_throws ArgumentError song_wang_diffusivity(pair, 298.0, 101.3e3)
        @test_throws ArgumentError song_wang_diffusivity(pair, 2001.0, 101.3e3)
    end
end

@testset "Müller--Klemm rows" begin
    @test length(MUELLER_KLEMM_1970) == 9
    @test MUELLER_KLEMM_TEMPERATURE == 273.15 + 24 && MUELLER_KLEMM_PRESSURE == 101325.0
    # Tab. 1 as printed, cm^2/s: the paper's own catarometric values at 1%
    # and the tritiated values it carries over at 2%.
    printed = Dict(("H2", "HD") => (1.349, 0.013, true),
                   ("H2", "D2") => (1.268, 0.013, true),
                   ("HD", "D2") => (1.126, 0.011, true),
                   ("H2", "HT") => (1.274, 0.025, false),
                   ("H2", "DT") => (1.212, 0.024, false),
                   ("H2", "T2") => (1.207, 0.030, false),
                   ("D2", "HT") => (1.044, 0.021, false),
                   ("D2", "DT") => (0.989, 0.020, false),
                   ("D2", "T2") => (0.956, 0.019, false))
    for ((a, b), (D, uncertainty, newly)) in printed
        pair = mueller_klemm_pair(b, a)
        @test pair.species == (a, b)
        @test pair.D ≈ D * 1e-4 rtol=2e-16
        @test pair.uncertainty ≈ uncertainty * 1e-4 rtol=2e-16
        @test pair.newly_measured == newly
        @test 0.008 < pair.uncertainty / pair.D < 0.026
        @test abs(pair.D_adjusted / pair.D - 1) < 0.015
    end
    @test mueller_klemm_pair("H2", "N2") === nothing
end

@testset "cross-source comparisons" begin
    T_mk, p_mk = MUELLER_KLEMM_TEMPERATURE, MUELLER_KLEMM_PRESSURE
    # Marrero and Mason's H2-D2 correlation against the Müller--Klemm
    # measurement it never used: 1.246 against 1.268 cm^2/s, inside the
    # paper's 2% group II limit at 300 K plus the measurement's 1%.
    mm = marrero_mason_diffusivity(marrero_mason_pair("H2", "D2"), T_mk, p_mk)
    mk = mueller_klemm_pair("H2", "D2")
    @test abs(mm / mk.D - 1) < 0.02 + mk.uncertainty / mk.D
    @test abs(mm / mk.D - 1) > 0.01

    # Every Müller--Klemm pair against the Song--Wang calculation at the
    # calculation's lowest tabulated temperature, 1 K above the
    # measurement, which moves D by under 0.6%.
    for mk in MUELLER_KLEMM_1970
        sw = song_wang_pair(mk.species...)
        @test sw !== nothing || mk.species == ("H2", "D2")
        sw === nothing && continue
        calculated = song_wang_diffusivity(sw, 298.15, p_mk)
        @test abs(calculated / mk.D - 1) <
              mk.uncertainty / mk.D + SONG_WANG_UNCERTAINTY + 0.006
    end

    # Marrero and Mason's evaluated He-H2 against the calculated 4He-H2.
    mm_he = marrero_mason_diffusivity(marrero_mason_pair("He", "H2"), 298.15, 101.3e3)
    sw_he = song_wang_diffusivity(song_wang_pair("4He", "H2"), 298.15, 101.3e3)
    @test abs(mm_he / sw_he - 1) < 0.02 + SONG_WANG_UNCERTAINTY
end

@testset "mixed-source hydrogen isotopologue fixture" begin
    species = ("H2", "HD", "D2")
    @test_throws ArgumentError neutral_binary_diffusion(species)
    sources = neutral_binary_sources(species; source=(:marrero_mason, :song_wang))
    @test sources[("H2", "D2")] === :marrero_mason
    @test sources[("H2", "HD")] === :song_wang && sources[("HD", "D2")] === :song_wang
    @test neutral_binary_sources(species)[("H2", "HD")] === nothing
    @test_throws ArgumentError neutral_binary_sources(species; source=:unknown)

    # A single-temperature measurement is not eligible at the default 300 K:
    # the next preferred source supplies the pair instead.
    fallback_sources = neutral_binary_sources(("H2", "HD");
                                               source=(:mueller_klemm, :song_wang))
    @test fallback_sources[("H2", "HD")] === :song_wang
    fallback = neutral_binary_diffusion(("H2", "HD");
                                        source=(:mueller_klemm, :song_wang))
    @test binary_diffusivity(fallback, 300.0, 101325.0, 1, 2) ≈
          song_wang_diffusivity(song_wang_pair("H2", "HD"), 300.0, 101325.0) rtol=2e-15
    @test neutral_binary_sources(("H2", "D2");
                                 source=(:mueller_klemm, :marrero_mason))[("H2", "D2")] ===
          :marrero_mason
    mk_sources = neutral_binary_sources(("H2", "HD"); source=(:song_wang, :mueller_klemm),
                                        temperature_ref=MUELLER_KLEMM_TEMPERATURE)
    @test mk_sources[("H2", "HD")] === :mueller_klemm
    @test neutral_binary_sources(("H2", "HD"); source=:song_wang,
                                 temperature_ref=MUELLER_KLEMM_TEMPERATURE)[("H2", "HD")] ===
          nothing

    model = neutral_binary_diffusion(species; source=(:marrero_mason, :song_wang))
    @test temperature_domain(model) == (298.15, 2000.0)
    @test binary_diffusivity(model, 300.0, 101325.0, 1, 3) ==
          marrero_mason_diffusivity(marrero_mason_pair("H2", "D2"), 300.0, 101325.0)
    for (i, j, a, b) in ((1, 2, "H2", "HD"), (2, 3, "HD", "D2"))
        pair = song_wang_pair(a, b)
        for T in (298.15, 700.0, 2000.0)
            @test binary_diffusivity(model, T, 101325.0, i, j) ≈
                  song_wang_diffusivity(pair, T, 101325.0) rtol=1e-12
        end
        residual = neutral_binary_diffusion_residual(model, i, j;
                                                     source=(:marrero_mason, :song_wang))
        @test pair.node_residual <= residual < 1e-4
    end
    @test_throws ArgumentError neutral_binary_diffusion_residual(model, 1, 2)

    # Re-centring on another reference temperature and padding to another
    # degree reproduce the same function; degree below four is refused.
    shifted = neutral_binary_diffusion(species; source=(:marrero_mason, :song_wang),
                                       temperature_ref=1000.0, degree=6,
                                       pressure_ref=2.0e5)
    for T in (298.15, 450.0, 2000.0)
        @test binary_diffusivity(shifted, T, 101325.0, 1, 2) ≈
              binary_diffusivity(model, T, 101325.0, 1, 2) rtol=1e-12
    end
    @test_throws ArgumentError neutral_binary_diffusion(species; degree=3,
                                                         source=(:marrero_mason, :song_wang))

    # Every calculated pair, with tritium, builds and matches its source;
    # H2-D2 and 3He-4He, which the calculation omits, fall through to
    # Marrero and Mason and carry that source's fit residual.
    isotopes = ("H2", "HD", "HT", "D2", "DT", "T2", "3He", "4He")
    @test_throws ArgumentError neutral_binary_diffusion(isotopes; source=:song_wang)
    tritium = neutral_binary_diffusion(isotopes; source=(:song_wang, :marrero_mason),
                                       temperature_ref=500.0)
    chosen = neutral_binary_sources(isotopes; source=(:song_wang, :marrero_mason))
    @test Set(k for (k, v) in chosen if v === :marrero_mason) ==
          Set([("H2", "D2"), ("3He", "4He")])
    names = species_names(tritium)
    for i in 1:8, j in i+1:8
        residual = neutral_binary_diffusion_residual(tritium, i, j;
                                                     source=(:song_wang, :marrero_mason))
        @test residual < (chosen[(names[i], names[j])] === :song_wang ? 1e-4 : 1e-3)
    end

    # The measured source is a single-temperature model.
    measured = neutral_binary_diffusion(species; source=:mueller_klemm,
                                        temperature_ref=MUELLER_KLEMM_TEMPERATURE)
    T_mk, p_mk = MUELLER_KLEMM_TEMPERATURE, MUELLER_KLEMM_PRESSURE
    @test temperature_domain(measured) == (T_mk, T_mk)
    @test binary_diffusivity(measured, T_mk, 2p_mk, 1, 2) ==
          mueller_klemm_pair("H2", "HD").D / 2
    @test neutral_binary_diffusion_residual(measured, 2, 3; source=:mueller_klemm) < 1e-14
    @test_throws ArgumentError binary_diffusivity(measured, 300.0, 101325.0, 1, 2)
    @test_throws ArgumentError neutral_binary_diffusion(species; source=:mueller_klemm)
    @test neutral_binary_sources(species; source=(:mueller_klemm, :song_wang),
                                 temperature_ref=MUELLER_KLEMM_TEMPERATURE)[("H2", "HD")] ===
          :mueller_klemm
end
