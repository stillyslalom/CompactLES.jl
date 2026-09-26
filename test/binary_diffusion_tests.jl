# Synthetic contract tests for pair-specific binary-diffusion polynomials.
# These coefficients exercise the API algebra; they are not validated H/D/T data.
# Included by runtests.jl and runnable directly.

using CompactLES
using CompactLES.DiffusionData
using Test

function synthetic_binary_diffusion(::Type{T}=Float64) where {T<:AbstractFloat}
    species = ("H2", "D2", "T2")
    D = T[0 1.0e-4 2.0e-4;
          1.0e-4 0 3.0e-4;
          2.0e-4 3.0e-4 0]
    coefficients = zeros(T, 3, 3, 2)
    coefficients[:, :, 1] .= T[0 1.0 1.5; 1.0 0 2.0; 1.5 2.0 0]
    coefficients[:, :, 2] .= T[0 0.10 -0.05; 0.10 0 0.20; -0.05 0.20 0]
    Tmin = T[0 200 250; 200 0 300; 250 300 0]
    Tmax = T[0 900 1000; 900 0 1100; 1000 1100 0]
    model = BinaryDiffusionPolynomial(species, D, coefficients, Tmin, Tmax;
                                      temperature_ref=T(300), pressure_ref=T(1.0e5))
    return (; model, species, D, coefficients, Tmin, Tmax)
end

sm_interdiffusivity(args...) = stanton_murillo_interdiffusivity(
    args...; assume_fully_ionized=true)

@testset "Stanton--Murillo fully ionized reference limit" begin
    masses = (H_ION_MASS, D_ION_MASS, T_ION_MASS)
    fractions = (0.2, 0.3, 0.5)
    rho, temperature = 100.0, 1.0e7
    hd = @inferred sm_interdiffusivity(
        masses, fractions, rho, temperature, 1, 2)

    # The density conversion is independent of the collision-fit evaluation.
    @test hd.diagnostics.n ≈ 2.6033003277535226e28 rtol=3e-14
    @test hd.diagnostics.assumptions === :fully_ionized_equilibrium
    @test hd.diagnostics.theta > 0
    @test all(value -> value > 0, hd.diagnostics.gamma)
    @test hd.diagnostics.ne == hd.diagnostics.n

    # Independent outputs from the authors' SMT EFF calculator, retrieved
    # 2026-09-18 from
    # https://tempest-stc.msu.edu/sites/default/files/tools/smt-transport%281%29.html
    # with EFF transport, fully ionized Z=1, and finite-temperature
    # Thomas--Fermi electron screening. Its older physical constants account
    # for the remaining O(1e-9) relative difference.
    oracle_temperature = 1000 * 1.602176634e-19 / 1.380649e-23 # 1000 eV in K
    oracle_cases = (
        ((0.5, 0.5, 0.0), 1, 2, 3.68705342575712e-3,
         0.016853405572343914, 0.9302753593378538),
        ((0.5, 0.0, 0.5), 1, 3, 4.46417121167910e-3,
         0.01463841830587654, 0.9648316721938148),
        ((0.0, 0.5, 0.5), 2, 3, 4.29312041614174e-3,
         0.013113881716292044, 0.9918650420922253),
    )
    for (x, i, j, expected_D, expected_coupling, expected_K11) in oracle_cases
        got = sm_interdiffusivity(masses, x, 1.0e4, oracle_temperature, i, j)
        @test got.D ≈ expected_D rtol=1e-8
        @test got.diagnostics.coupling ≈ expected_coupling rtol=1e-8
        @test got.diagnostics.K11 ≈ expected_K11 rtol=1e-8
        @test got.diagnostics.electron_screening_energy >= 2got.diagnostics.Fermi_energy / 3
    end
    off_reference = sm_interdiffusivity(
        masses, (0.0, 0.2, 0.8), 1.0e3,
        2000 * 1.602176634e-19 / 1.380649e-23, 2, 3)
    @test off_reference.D ≈ 0.174438236578357 rtol=1e-8

    dh = sm_interdiffusivity(masses, fractions, rho, temperature, 2, 1)
    @test dh.D == hd.D
    @test dh.diagnostics == hd.diagnostics

    permutation = (3, 1, 2)
    permuted_masses = ntuple(k -> masses[permutation[k]], 3)
    permuted_fractions = ntuple(k -> fractions[permutation[k]], 3)
    permuted = sm_interdiffusivity(
        permuted_masses, permuted_fractions, rho, temperature, 2, 3)
    @test permuted.D == hd.D
    @test permuted.diagnostics.n == hd.diagnostics.n
    @test permuted.diagnostics.lambda_eff == hd.diagnostics.lambda_eff

    # A trace ion makes only its own Debye length infinite. The binary pair
    # coefficient remains finite because total ion density sets Eq. 65.
    trace = sm_interdiffusivity(
        masses, (0.5, 0.5, 0.0), rho, temperature, 1, 3)
    @test isfinite(trace.D) && trace.D > 0
    @test isinf(trace.diagnostics.lambda_i[3])
    @test all(isfinite, trace.diagnostics.lambda_i[1:2])

    # Directly cover the two published K11 fits and their rounded join.
    weak_g = 1.0e-8
    @test CompactLES.DiffusionData._sm_K11(weak_g) ≈ -log(1.4660weak_g) / 4 rtol=2e-8
    below = CompactLES.DiffusionData._sm_K11(prevfloat(1.0))
    above = CompactLES.DiffusionData._sm_K11(1.0)
    @test below > 0 && above > 0
    @test below ≈ above rtol=2e-4

    result32 = @inferred sm_interdiffusivity(
        Float32.(masses), (0.2f0, 0.3f0, 0.5f0), 100.0f0, 1.0f7, 1, 2)
    @test result32.D isa Float64
    @test result32.diagnostics.theta isa Float64

    @test_throws ArgumentError stanton_murillo_interdiffusivity(
        masses, fractions, rho, temperature, 1, 2)
    @test_throws ArgumentError sm_interdiffusivity(masses, fractions, rho, 300.0, 1, 2)

    bad_calls = (
        (() -> sm_interdiffusivity((H_ION_MASS,), (1.0,), rho, temperature, 1, 1)),
        (() -> sm_interdiffusivity(masses, fractions, 0.0, temperature, 1, 2)),
        (() -> sm_interdiffusivity(masses, fractions, rho, NaN, 1, 2)),
        (() -> sm_interdiffusivity((0.0, D_ION_MASS, T_ION_MASS), fractions, rho, temperature, 1, 2)),
        (() -> sm_interdiffusivity(masses, (-0.1, 0.6, 0.5), rho, temperature, 1, 2)),
        (() -> sm_interdiffusivity(masses, (0.2, 0.3, 0.4), rho, temperature, 1, 2)),
        (() -> sm_interdiffusivity(masses, fractions, rho, temperature, 0, 2)),
        (() -> sm_interdiffusivity(masses, fractions, rho, temperature, 2, 2)),
    )
    for call in bad_calls
        @test_throws ArgumentError call()
    end
end

@testset "synthetic binary-diffusion polynomial" begin
    fixture = synthetic_binary_diffusion()
    model = fixture.model
    @test isbitstype(typeof(model))
    @test species_names(model) == collect(fixture.species)

    # At the reference temperature and pressure the polynomial factor is one.
    @test binary_diffusivity(model, 300.0, 1.0e5, 1, 2) ≈ fixture.D[1, 2] rtol=2e-15
    @test binary_diffusivity(model, 300.0, 2.0e5, 1, 2) ≈ fixture.D[1, 2] / 2 rtol=2e-15

    temperature, pressure = 600.0, 2.5e5
    z = log(temperature / 300.0)
    expected = fixture.D[2, 3] * exp(2.0z + 0.20z^2) * 1.0e5 / pressure
    @test binary_diffusivity(model, temperature, pressure, 2, 3) ≈ expected rtol=2e-15
    @test binary_diffusivity(model, temperature, pressure, 3, 2) ==
          binary_diffusivity(model, temperature, pressure, 2, 3)

    # Reordering species and every pair table preserves species-pair values.
    permutation = (3, 1, 2)
    reordered = BinaryDiffusionPolynomial(
        ntuple(i -> fixture.species[permutation[i]], 3),
        fixture.D[[permutation...], [permutation...]],
        fixture.coefficients[[permutation...], [permutation...], :],
        fixture.Tmin[[permutation...], [permutation...]],
        fixture.Tmax[[permutation...], [permutation...]];
        temperature_ref=300.0, pressure_ref=1.0e5)
    @test binary_diffusivity(reordered, temperature, pressure, 3, 1) ==
          binary_diffusivity(model, temperature, pressure, 2, 3)

    for pair in ((1, 2), (1, 3), (2, 3))
        i, j = pair
        @test isfinite(binary_diffusivity(model, fixture.Tmin[i, j], pressure, i, j))
        @test isfinite(binary_diffusivity(model, fixture.Tmax[i, j], pressure, i, j))
        @test_throws ArgumentError binary_diffusivity(model, prevfloat(fixture.Tmin[i, j]), pressure, i, j)
        @test_throws ArgumentError binary_diffusivity(model, nextfloat(fixture.Tmax[i, j]), pressure, i, j)
    end
    for (temperature_bad, pressure_bad, i, j) in
            ((0.0, pressure, 1, 2), (-1.0, pressure, 1, 2),
             (NaN, pressure, 1, 2), (temperature, 0.0, 1, 2),
             (temperature, -1.0, 1, 2), (temperature, Inf, 1, 2),
             (temperature, pressure, 0, 2), (temperature, pressure, 1, 4),
             (temperature, pressure, 2, 2))
        @test_throws ArgumentError binary_diffusivity(model, temperature_bad, pressure_bad, i, j)
    end
end

@testset "binary-diffusion polynomial construction and scalar type" begin
    fixture32 = synthetic_binary_diffusion(Float32)
    model32 = fixture32.model
    value32 = @inferred binary_diffusivity(model32, 600.0f0, 2.5f5, 1, 3)
    @test value32 isa Float32
    @test isbitstype(typeof(model32))
    # Validate in the caller's precision before conversion: this Float64 value
    # would otherwise round back onto the Float32 inclusive upper boundary.
    @test_throws ArgumentError binary_diffusivity(
        model32, nextfloat(Float64(fixture32.Tmax[1, 2])), 2.5f5, 1, 2)

    good = synthetic_binary_diffusion()
    D, C = good.D, good.coefficients
    Tmin, Tmax = good.Tmin, good.Tmax
    @test_throws ArgumentError BinaryDiffusionPolynomial(String[], zeros(0, 0), zeros(0, 0, 1), zeros(0, 0), zeros(0, 0))
    @test_throws ArgumentError BinaryDiffusionPolynomial(("H", "H"), D[1:2, 1:2], C[1:2, 1:2, :], Tmin[1:2, 1:2], Tmax[1:2, 1:2])
    @test_throws ArgumentError BinaryDiffusionPolynomial(("H", "D"), D, C, Tmin, Tmax)
    @test_throws ArgumentError BinaryDiffusionPolynomial(("H", "D", "T"), D, zeros(3, 3, 0), Tmin, Tmax)

    asymmetric_D = copy(D); asymmetric_D[1, 2] *= 2
    asymmetric_C = copy(C); asymmetric_C[1, 2, 1] += 1
    asymmetric_min = copy(Tmin); asymmetric_min[1, 2] += 1
    invalid_D = copy(D); invalid_D[1, 2] = invalid_D[2, 1] = 0
    invalid_range = copy(Tmax); invalid_range[1, 2] = invalid_range[2, 1] = 250
    for args in ((asymmetric_D, C, Tmin, Tmax), (D, asymmetric_C, Tmin, Tmax),
                 (D, C, asymmetric_min, Tmax), (invalid_D, C, Tmin, Tmax),
                 (D, C, Tmin, invalid_range))
        @test_throws ArgumentError BinaryDiffusionPolynomial((:H, :D, :T), args...)
    end
    @test_throws ArgumentError BinaryDiffusionPolynomial((:H, :D, :T), D, C, Tmin, Tmax;
                                                          temperature_ref=0.0)
    @test_throws ArgumentError BinaryDiffusionPolynomial((:H, :D, :T), D, C, Tmin, Tmax;
                                                          pressure_ref=Inf)
end
