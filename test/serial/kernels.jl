# The serial suite's tests of script arguments, the pointwise kernels and the device
# line solves. Included by test/serial_suite.jl, which loads the package and defines
# the imports and helpers used here; not runnable on its own.

# Not solver code, but `bench/` and the cluster scripts are not imported by this
# suite, so if these are not tested here they are not tested anywhere — and the
# whole reason they exist is to turn a silent wrong-default run into an error.
@testset "script argument parsing" begin
    defaults = (N=32, tfinal=10.0, configs="off:1", progress=0, quiet=false)
    pos = (:N, :tfinal)

    @test script_args(String[], defaults; positional=pos) == defaults
    opt = script_args(["128", "2.5", "progress=200", "configs=on:1:0.008"],
                      defaults; positional=pos)
    @test (opt.N, opt.tfinal, opt.progress) == (128, 2.5, 200)
    @test opt.configs == "on:1:0.008"      # String defaults pass through verbatim
    # Order is free, and a positional may be supplied by name instead.
    @test script_args(["progress=5", "64"], defaults; positional=pos).N == 64
    @test script_args(["N=64"], defaults; positional=pos).N == 64
    @test script_args(["nmax=1"], (nmax=typemax(Int),)).nmax == 1

    for text in ("true", "yes", "on", "1")
        @test script_args(["quiet=$text"], defaults; positional=pos).quiet
    end
    for text in ("false", "no", "off", "0")
        @test !script_args(["quiet=$text"], defaults; positional=pos).quiet
    end

    # The point of the exercise: every one of these would be a silent default
    # in a parser that falls through on an unknown name.
    @test_throws ArgumentError script_args(["progerss=1"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["progress=lots"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["quiet=maybe"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["1", "2", "3"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["64"], defaults)
    @test_throws ArgumentError script_args(String[], defaults; positional=(:nope,))

    @test script_grid("128") == (128, 128, 128)
    @test script_grid("256,256,512") == (256, 256, 512)
    @test_throws ArgumentError script_grid("128,256")
    @test_throws ArgumentError script_grid("128x128")
end

@testset "pointwise kernels: the KA path reproduces the threaded path" begin
    # Every pointwise phase is one shared per-point
    # body behind two launchers. The bodies are per-point independent, so the
    # KernelAbstractions CPU path must reproduce the @threaded path BITWISE —
    # not to round-off — over a full run touching primitives, fluxes, sensors,
    # species diffusion, the RK update, and the state filter. Cylindrical
    # geometry in the second case adds the metric-correction and metric-source
    # bodies.
    per3k = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    function tube_pair()
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 0.6), (0.0, 1.0)), eos=eos,
                    bcs=((SlipWallBC(), SlipWallBC()), per3k[2], per3k[3]), ic=nothing),
            Numerics(n_global=(48, 32, 1)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.0))
        end)
        return s, Q
    end
    s1, Q1 = tube_pair()
    run!(s1, Q1; tfinal=0.02, nmax=50)
    local s2, Q2
    CL.with_knob(CL.FORCE_KA => true) do
        s2, Q2 = tube_pair()
        run!(s2, Q2; tfinal=0.02, nmax=50)
    end
    @test s1.step == s2.step
    @test parent(Q1) == parent(Q2)

    # The tube above runs the default partial-density species channel. The
    # same tube under the bulk channel adds the conserved-gradient flux body,
    # and under the Fickian one the per-species bound and diffusivity bodies
    # and the correction-velocity branch of the flux body.
    function channel_tube_pair(channel)
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 0.6), (0.0, 1.0)), eos=eos,
                    bcs=((SlipWallBC(), SlipWallBC()), per3k[2], per3k[3]), ic=nothing),
            Numerics(n_global=(48, 32, 1),
                     art=ArtificialProperties(enabled=true, species_flux=channel)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.0))
        end)
        return s, Q
    end
    for channel in (:bulk, :fickian)
        s1b, Q1b = channel_tube_pair(channel)
        run!(s1b, Q1b; tfinal=0.02, nmax=50)
        local s2b, Q2b
        CL.with_knob(CL.FORCE_KA => true) do
            s2b, Q2b = channel_tube_pair(channel)
            run!(s2b, Q2b; tfinal=0.02, nmax=50)
        end
        @test s1b.step == s2b.step
        @test parent(Q1b) == parent(Q2b)
        @test maximum(s1b.D_art[1]) > 0
        Ib = CL.interior(s1b.decomp)
        channel === :bulk && @test s1b.D_art[1][Ib] == s1b.D_art[2][Ib]
    end
    @test maximum(s1.D_art[1]) > 0
    @test s1.D_art[1][CL.interior(s1.decomp)] == s1.D_art[2][CL.interior(s1.decomp)]

    function axis_pair()
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)),
                    bcs=((AxisBC(), SlipWallBC()), per3k[2], per3k[3]),
                    metric=CylindricalMetric(), ic=nothing),
            Numerics(n_global=(32, 1, 24)))
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, z) ->
            Prim(u=(0.0, 0.2r, 0.05 * sin(2π * z)), p=1.0 + 0.02r^2, rho=1.0))
        return s, Q
    end
    s3, Q3 = axis_pair()
    run!(s3, Q3; tfinal=0.02, nmax=20)
    local s4, Q4
    CL.with_knob(CL.FORCE_KA => true) do
        s4, Q4 = axis_pair()
        run!(s4, Q4; tfinal=0.02, nmax=20)
    end
    @test s3.step == s4.step
    @test parent(Q3) == parent(Q4)
end

@testset "device line solves: DevicePlan reproduces the host plans" begin
    # The compact line solves run as
    # KernelAbstractions kernels in a (lines × n) layout, one thread per line,
    # with the reduced interface stage on the host. The device arithmetic
    # mirrors the host path per line operation for operation (including the
    # divide-vs-multiply-by-inverse split between the banded x and y/z
    # conventions), so on the KA CPU backend the comparison is BITWISE, the
    # same equality gate the pointwise kernels carry. Closed edges, folds
    # and periodic wrap all route through the same kernels; the fold cases
    # fill the halos with random data, since path equality does not require
    # physically meaningful mirror values.
    cpu = CL.KernelAbstractions.CPU()
    function compare(scheme, dim; periodic=false, lo_fold=nothing)
        # 13 is the smallest extent the six-row T8 closure set accepts.
        d = CL.Decomp((16, 13, 14), (periodic, periodic, periodic); dims=(1, 1, 1))
        plan = CL.plan_direction(d, scheme, dim, 0.1; lo_fold=lo_fold)
        dplan = device_plan(plan, cpu)
        f = CL.field(d)
        rand!(f)
        CL.exchange_halos!(f, d)
        out_h = CL.field(d)
        out_d = CL.field(d)
        apply_along!(out_h, plan, f, d)
        apply_along!(out_d, dplan, f, d)
        return view(out_h, CL.interior(d)) == view(out_d, CL.interior(d))
    end
    schemes = (lele_d1_6(Float64), lele_d1_6(Float64; closures=:brady_livescu),
               lele_d1_8(Float64), lele_d1_8(Float64; closures=:brady_livescu),
               lele_d1_10(Float64), compact_filter(0.45),
               gaussian_filter(Float64), compact_d8(Float64))
    for scheme in schemes, periodic in (false, true), dim in 1:3
        @test compare(scheme, dim; periodic=periodic)
    end
    for dim in 1:3, σ in (1, -1)
        @test compare(lele_d1_6(Float64), dim; lo_fold=σ)
    end

    # The allocation half of the backend interface: DeviceBackend routes field and
    # allocate_state through KernelAbstractions.zeros with the same halo
    # padding as the CPU backend.
    d = CL.Decomp((16, 12, 14), (false, true, true); dims=(1, 1, 1))
    bk = DeviceBackend(cpu)
    f = CL.field(bk, d)
    @test size(f) == size(CL.field(d))
    @test all(iszero, f)
    Q = allocate_state(bk, d, 7)
    @test Q isa ConservedState
    @test size(Q) == (size(CL.field(d))..., 7)
end
