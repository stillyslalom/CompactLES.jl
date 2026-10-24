# CompactLESBriareusExt: a Briareus CapsuleDrive read into an AblationDrive.
#
# Standalone, from an environment that develops both packages:
#   julia --project=<env> test/briareus_ext_tests.jl

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES, Briareus
using Test

@testset "Briareus extension" begin
    @test Base.get_extension(CompactLES, :CompactLESBriareusExt) !== nothing
    t = collect(0.0:0.05:2.0)
    Tr = 100 .+ 100 .* t
    flux = (Tr ./ 100) .^ 4 .* 1.0283                    # any positive flux
    modes = zeros(9, length(t))
    modes[1, :] .= 1
    modes[3, :] .= 0.1                                    # P2 = +0.1, pole-hot
    modes[5, :] .= -0.02
    cd = CapsuleDrive(t, Tr, flux, modes, fill(NaN, length(t)), 1.135, 16.2,
                      Dict(:code => "test"))
    d = radiation_drive(cd; peak=145.0, driver=3)
    @test d isa AblationDrive{8}
    @test d.Tr == Tr && d.time == t && d.driver == 3
    @test maximum(ablation_pressure(d, τ) for τ in t) ≈ 145.0
    @test CompactLES.drive_modes(d, 1.0)[2] ≈ 0.1 && CompactLES.drive_modes(d, 1.0)[4] ≈ -0.02
    @test CompactLES.drive_factor(d, 1.0, 1.0) > CompactLES.drive_factor(d, 0.0, 1.0)  # poles hotter
    @test radiation_drive(cd; lmax=0) isa AblationDrive{0}
    @test radiation_drive(cd; pressure=4.0).pressure == 4.0
    d2 = radiation_drive(cd; peak=100.0, t_on=0.5)
    @test maximum(ablation_pressure(d2, τ) for τ in t) ≈ 100.0
    @test_throws ArgumentError radiation_drive(cd; lmax=9)
    # every m (CapsuleDrive.ylm, Schmidt ratios): the drive's harmonics, m = 0 its modes,
    # and its angular factor is Briareus's Σ bₗₘ Ŷₗₘ (Ŷₗₘ = √(4π/(2l+1)) Yₗₘ, orthonormal Y)
    Y = zeros(81, length(t))
    Y[1, :] .= 1
    for l in 1:8
        Y[Briareus.mode_index(l, 0), :] .= modes[l+1, :]
    end
    Y[Briareus.mode_index(2, 2), :] .= 0.07
    Y[Briareus.mode_index(3, -1), :] .= -0.04
    Y[Briareus.mode_index(8, 5), :] .= 0.01
    cy = CapsuleDrive(t, Tr, flux, modes, fill(NaN, length(t)), 1.135, 16.2, Dict(:code => "test"), Y)
    dy = radiation_drive(cy; peak=145.0)
    @test dy isa AblationDrive{8} && size(dy.harmonics) == (81, length(t))
    @test CompactLES.drive_modes(dy, 1.0) == CompactLES.drive_modes(d, 1.0)
    @test isempty(radiation_drive(cy; azimuthal=false).harmonics)
    @test size(radiation_drive(cy; lmax=3).harmonics, 1) == 16
    B = zeros(9, 17)
    for (μ, φ) in ((0.3, 0.7), (-0.8, 2.5), (0.95, 5.0))
        Briareus._real_ylm!(B, 8, μ, φ)
        ref = sum(Y[Briareus.mode_index(l, m), 1] * sqrt(4π / (2l + 1)) * B[l+1, 9+m]
                  for l in 0:8 for m in -l:l)
        @test CompactLES.drive_factor(dy, μ, φ, 0.0) ≈ ref
    end
end
