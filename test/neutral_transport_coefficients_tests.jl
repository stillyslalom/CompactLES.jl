module NeutralTransportCoefficientTests

using CompactLES, Test

const CL = CompactLES

function polynomial_model(::Type{T}=Float64; exponent=T(1.75), Dscale=one(T)) where {T<:AbstractFloat}
    names = ("H2", "N2", "O2")
    D = T(Dscale) .* T[0 7.25e-5 8.10e-5;
                         7.25e-5 0 2.00e-5;
                         8.10e-5 2.00e-5 0]
    C = zeros(T, 3, 3, 1)
    for i in 1:3, j in 1:3
        i == j || (C[i, j, 1] = exponent)
    end
    Tmin = T[0 200 200; 200 0 200; 200 200 0]
    Tmax = T[0 1000 1000; 1000 0 1000; 1000 1000 0]
    return BinaryDiffusionPolynomial(names, D, C, Tmin, Tmax;
                                     temperature_ref=T(300), pressure_ref=T(101325))
end

@testset "polynomial CEA transport construction and scalar contract" begin
    eos = Nasa9Mixture(["H2", "N2", "O2"])
    poly = polynomial_model()
    tr = CeaTransport(eos; diffusion=:mixture_averaged, binary_diffusion=poly)
    @test CL.transport_has_domain(tr)
    @test tr.diffusion isa BinaryDiffusionPolynomial{Float64,3}

    # Pair labels are part of the model identity, not merely documentation.
    reversed = BinaryDiffusionPolynomial(("N2", "H2", "O2"),
        [0.0 7.25e-5 2.00e-5; 7.25e-5 0 8.10e-5; 2.00e-5 8.10e-5 0],
        reshape([0.0, 1.75, 1.75, 1.75, 0.0, 1.75, 1.75, 1.75, 0.0], 3, 3, 1),
        [0.0 200 200; 200 0 200; 200 200 0],
        [0.0 1000 1000; 1000 0 1000; 1000 1000 0])
    @test_throws ArgumentError CeaTransport(eos; diffusion=:mixture_averaged,
                                            binary_diffusion=reversed)
    one = BinaryDiffusionPolynomial(("H2",), zeros(1, 1), zeros(1, 1, 1),
                                    zeros(1, 1), zeros(1, 1))
    @test_throws ArgumentError CeaTransport(eos; diffusion=:mixture_averaged,
                                            binary_diffusion=one)

    eos32 = Nasa9Mixture(Float32, ("H2", "N2", "O2"))
    tr32 = CeaTransport(eos32; diffusion=:mixture_averaged,
                         binary_diffusion=polynomial_model())
    @test tr32.diffusion isa BinaryDiffusionPolynomial{Float32,3}
    @test CL.transport_domain_status(tr32, eos32, 300f0, 1f0,
                                     (0.2f0, 0.3f0, 0.5f0)) == CL.TRANSPORT_OK
    # A source model can be well formed in Float64 yet lose a required finite,
    # positive value when CeaTransport adopts the EOS scalar type.
    for invalid in (polynomial_model(Dscale=1e-296), polynomial_model(Dscale=1e304),
                    polynomial_model(exponent=1e300))
        @test_throws ArgumentError CeaTransport(eos32; diffusion=:mixture_averaged,
                                                binary_diffusion=invalid)
    end
    # The comparison precedes conversion to Float32, preserving the source
    # endpoint even if the caller supplies a wider scalar.
    @test CL.transport_domain_status(tr32, eos32,
        nextfloat(Float64(tr32.diffusion.temperature_max[1][2])), 1.0,
        (0.2, 0.3, 0.5)) == CL.TRANSPORT_TEMPERATURE_OUT_OF_RANGE

    # Integer literals follow the same original-precision range policy.  At
    # this Float32 endpoint, converting 2^24 + 1 first would round down.
    bound = Float32(2^24)
    Dbound = Float32[0 7.25e-5 8.10e-5;
                     7.25e-5 0 2.00e-5;
                     8.10e-5 2.00e-5 0]
    Cbound = zeros(Float32, 3, 3, 1)
    for i in 1:3, j in 1:3
        i == j || (Cbound[i, j, 1] = 1.75f0)
    end
    Tminbound = Float32[0 200 200; 200 0 200; 200 200 0]
    Tmaxbound = Float32[0 bound bound; bound 0 bound; bound bound 0]
    bounded = CeaTransport(eos32; diffusion=:mixture_averaged,
        binary_diffusion=BinaryDiffusionPolynomial(("H2", "N2", "O2"), Dbound,
            Cbound, Tminbound, Tmaxbound))
    @test_throws ArgumentError transport_coefficients(bounded, eos32, 2^24 + 1,
                                                       1f0, 1200f0,
                                                       (0.2f0, 0.3f0, 0.5f0))
end

@testset "polynomial closure matches legacy binary diffusion" begin
    eos = Nasa9Mixture(["H2", "N2", "O2"])
    poly = polynomial_model()
    direct = BinaryDiffusion([0.0 7.25e-5 8.10e-5;
                              7.25e-5 0 2.00e-5;
                              8.10e-5 2.00e-5 0])
    polynomial_transport = CeaTransport(eos; diffusion=:mixture_averaged,
                                         binary_diffusion=poly)
    direct_transport = CeaTransport(eos; diffusion=:mixture_averaged,
                                     binary_diffusion=direct)
    temperature, pressure = 620.0, 2.3e5
    for Y in ((0.2, 0.3, 0.5), (0.9, 0.09, 0.01), (0.0, 0.4, 0.6))
        R = Tuple(eos.Rk)
        rho = pressure / (temperature * sum(Y[k] * R[k] for k in 1:3))
        cp = sum(Y[k] * CL.species_cp(eos, k, temperature) for k in 1:3)
        got = transport_coefficients(polynomial_transport, eos, temperature, rho, cp, Y)
        expected = transport_coefficients(direct_transport, eos, temperature, rho, cp, Y)
        @test all(isapprox(got.D[k], expected.D[k]; rtol=3e-13) for k in 1:3)
    end
    scale = (temperature / 300)^1.75 * 101325 / pressure
    @test isapprox(binary_diffusivity(poly, temperature, pressure, 1, 3),
                   8.10e-5 * scale; rtol=3e-13)
    @test isapprox(binary_diffusivity(poly, temperature, pressure, 2, 3),
                   2.00e-5 * scale; rtol=3e-13)
end

@testset "polynomial transport rejects invalid scalar states" begin
    eos = Nasa9Mixture(["H2", "N2", "O2"])
    tr = CeaTransport(eos; diffusion=:mixture_averaged,
                      binary_diffusion=polynomial_model())
    Y, cp = (0.2, 0.3, 0.5), 1200.0
    @test_throws ArgumentError transport_coefficients(tr, eos, 199.0, 1.0, cp, Y)
    @test_throws ArgumentError transport_coefficients(tr, eos, 300.0, 0.0, cp, Y)
    @test_throws ArgumentError transport_coefficients(tr, eos, NaN, 1.0, cp, Y)

    # Pair evaluation itself is part of the domain: a finite source fit can
    # overflow or underflow before reaching its stated temperature endpoint.
    for exponent in (2000.0, -2000.0)
        overflow = CeaTransport(eos; diffusion=:mixture_averaged,
                                binary_diffusion=polynomial_model(exponent=exponent))
        @test CL.transport_domain_status(overflow, eos, 600.0, 1.0, Y) ==
              CL.TRANSPORT_INVALID_COEFFICIENT
        @test_throws ArgumentError transport_coefficients(overflow, eos, 600.0, 1.0, cp, Y)
    end
end

end # module
