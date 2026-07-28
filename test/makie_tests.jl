# Makie extension tests. Included by runtests.jl when a Makie backend is
# loadable, and runnable under mpiexec for the decomposition-independent
# profile, which is the property `line_profile` has and the old rank-local
# `density_line` did not.
#
#   julia --project=docs test/makie_tests.jl               # serial
#   mpiexec -n 4 julia --project=docs test/makie_tests.jl  # decomposed
#
# A Makie backend is a weak dependency, so it is not loadable from the package
# environment alone. Run these from an environment that has both — the docs
# environment carries CairoMakie.

if !@isdefined(CL)
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Test
    const CL = CompactLES
end
using CairoMakie

# The old `density_line`: rank-local sampling of mixture_density along dim 1 at
# (i, 1, 1). Kept here as the reference the new API must reproduce.
function density_line_local(solver, Q)
    n = solver.decomp.n_local[1]
    [mixture_density(solver, Q, gidx(solver, i, 1, 1)) for i in 1:n]
end

@testset "Makie extension: extraction API" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    @test makie_available()

    # A 1-D acoustic-style setup: density varies along x, other dims collapsed.
    gamma = 1.4
    prob = Problem(name="viz", eos=single_species(gamma=gamma),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
                   ic=(x, y, z) -> begin
                       p = 1 + 0.1 * exp(-((x - 0.4) / 0.05)^2)
                       Prim(p=p, rho=p^(1 / gamma))
                   end)
    num = Numerics(n_global=(96, 1, 1), art=ArtParams(enabled=false),
                   filter_interval=0, dims=(np, 1, 1))
    solver, Q = setup(prob, num)

    # line_profile is the global, gathered replacement. Its value equals the
    # concatenation of the per-rank local samples in rank order, so on one rank
    # it is exactly the old density_line, and on many it is the whole line —
    # which the rank-local helper could not produce.
    coord, value = line_profile(solver, Q, :rho)
    @test length(coord) == 96
    @test length(value) == 96
    @test issorted(coord)

    local_rho = density_line_local(solver, Q)
    counts = MPI.Allgather(Cint(length(local_rho)), comm)
    gathered = Vector{Float64}(undef, sum(counts))
    MPI.Allgatherv!(local_rho, MPI.VBuffer(gathered, counts), comm)
    # Transverse dims are collapsed, so the plane average is the point value:
    # the gathered per-rank line equals the global profile bit for bit.
    @test value == gathered

    # The profile is identical on every rank (a collective reduction, not a
    # rank-local read). Compare against rank 0's copy.
    ref = MPI.bcast(value, comm; root=0)
    @test value == ref
end

@testset "Makie extension: slices and revolution" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)

    # A 2-D Cartesian field, decomposed along x when ranks allow it. 48 along x
    # keeps ≥ 9 points per rank up to np = 4, which the C8 filter closure needs.
    dims = np == 1 ? (1, 1, 1) : (np, 1, 1)
    prob = Problem(name="slice", eos=single_species(gamma=1.4),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
                   ic=(x, y, z) -> Prim(p=1.0,
                                        rho=1.0 + 0.5sin(2pi * x) * cos(2pi * y)))
    num = Numerics(n_global=(48, 16, 1), art=ArtParams(enabled=false),
                   filter_interval=0, dims=dims)
    solver, Q = setup(prob, num)

    slice = field_slice(solver, Q, :rho; normal=3, index=1)
    if MPI.Comm_rank(comm) == 0
        x1, x2, vals = slice
        @test size(vals) == (48, 16)
        @test length(x1) == 48 && length(x2) == 16
        # The analytic field is recovered on the gathered global plane.
        @test isapprox(vals[6, 4], 1.0 + 0.5sin(2pi * x1[6]) * cos(2pi * x2[4]);
                       atol=1e-12)
        X, Y, grid = cartesian_slice(solver, (1, 2), x1, x2, vals; n=32)
        @test size(grid) == (32, 32)
        @test all(isfinite, grid)           # Cartesian slice fills the raster
    else
        @test slice === nothing             # a slice is a rank-0 gather
    end

    # revolve_profile is the collapsed-radial view: a monotone bump becomes a
    # disk whose finite fraction is the inscribed area π/4.
    r = collect(range(0.02, 1.0; length=40))
    v = exp.(-((r .- 0.5) ./ 0.1) .^ 2)
    axis, disk = revolve_profile(r, v; n=120)
    @test size(disk) == (120, 120)
    @test isapprox(count(isfinite, disk) / length(disk), pi / 4; atol=0.03)
end

# Building a plot is a rank-0 / serial concern: `profileplot` returns a replicated
# figure and `fieldheatmap` gathers to rank 0. The curvilinear solver here also
# resolves θ, which does not decompose cleanly at small extents, so this runs
# only in serial.
if MPI.Comm_size(MPI.COMM_WORLD) == 1
    @testset "Makie extension: plot objects" begin
        prob = Problem(name="plot", eos=single_species(gamma=1.4),
                       metric=CylindricalMetric(),
                       domain=((0.0, 1.0), (0.0, 2pi), (0.0, 1.0)),
                       bcs=((AxisBC(), SlipWallBC()),
                            (PeriodicBC(), PeriodicBC()),
                            (PeriodicBC(), PeriodicBC())),
                       ic=(r, th, z) -> Prim(p=1.0, rho=1.0 + 0.2cos(th) * r))
        num = Numerics(n_global=(24, 16, 1), art=ArtParams(enabled=false),
                       filter_interval=0)
        solver, Q = setup(prob, num)

        fig, ax, plt = profileplot(solver, Q, :rho)
        @test fig isa Makie.Figure
        @test plt isa Makie.Lines

        fig2, ax2, plt2 = fieldheatmap(solver, Q, :rho; normal=3, index=1)
        @test fig2 isa Makie.Figure
        @test plt2 isa Makie.Heatmap

        # A spherical meridian (normal = 3 → (r, θ) plane) resamples onto x–z with
        # the pole vertical, not x–y. Slicing the 3-D position by (a, b) collapsed
        # the plane onto Y = 0 and rastered to all-NaN; guard that the meridian
        # fills a physical half-disk (≈ π/4 of the bounding box).
        sph = Problem(name="meridian", eos=single_species(gamma=1.4),
                      metric=SphericalMetric(),
                      domain=((0.0, 1.0), (0.0, pi), (0.0, 2pi)),
                      bcs=((OriginBC(), SlipWallBC()),
                           (PoleBC(), PoleBC()),
                           (PeriodicBC(), PeriodicBC())),
                      ic=(r, th, ph) -> Prim(p=1.0, rho=1.0 + exp(-(r / 0.25)^2)))
        snum = Numerics(n_global=(24, 16, 12), art=ArtParams(enabled=false),
                        filter_interval=0)
        ssolver, sQ = setup(sph, snum)
        x1, x2, vals = field_slice(ssolver, sQ, :rho; normal=3, index=1)
        X, Y, grid = cartesian_slice(ssolver, (1, 2), x1, x2, vals; n=100)
        finite = count(isfinite, grid) / length(grid)
        @test finite > 0.5                       # not the degenerate all-NaN case
        @test isapprox(finite, pi / 4; atol=0.06)
    end
end

println("Makie extension tests complete")
