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
end
