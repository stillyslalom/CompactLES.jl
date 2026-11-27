# The serial suite's tests of the diagnostics and the field output. Included by
# test/serial_suite.jl, which loads the package and defines the imports and helpers
# used here; not runnable on its own.

@testset "line_sample: one grid line, distinct from the plane average" begin
    # A density linear in all three coordinates separates the two extractions:
    # a sample along x at (j, k) carries 0.2 y_j + 0.3 z_k, the plane average
    # carries the transverse means instead, and every value is definitional.
    ic(x, y, z) = Prim(p=1.0, rho=1 + 0.1x + 0.2y + 0.3z)
    rho_fn(x, y, z) = 1 + 0.1x + 0.2y + 0.3z
    solver = build_solver(
        Problem(domain=((0.0, 2.0), (0.0, 1.0), (0.0, 0.5)), bcs=per3, ic=nothing),
        Numerics(n_global=(24, 16, 12), art=ArtificialProperties(enabled=false)))
    Q = allocate_state(solver)
    initialize!(solver, Q, ic)
    gx(d, g) = global_xcoord(solver, d, g)

    coord, value = line_sample(solver, Q, :rho; dim=1, index=(7, 4))
    @test coord == [gx(1, g) for g in 1:24]
    @test maximum(abs, value .- [rho_fn(gx(1, g), gx(2, 7), gx(3, 4)) for g in 1:24]) < 1e-14
    coord, value = line_sample(solver, Q, :rho; dim=2, index=(3, 11))
    @test coord == [gx(2, g) for g in 1:16]
    @test maximum(abs, value .- [rho_fn(gx(1, 3), gx(2, g), gx(3, 11)) for g in 1:16]) < 1e-14
    coord, value = line_sample(solver, Q, :rho; dim=3, index=(24, 16))
    @test coord == [gx(3, g) for g in 1:12]
    @test maximum(abs, value .- [rho_fn(gx(1, 24), gx(2, 16), gx(3, g)) for g in 1:12]) < 1e-14

    # The default line is (i, 1, 1); `at` snaps each transverse coordinate to
    # its nearest node, here (7, 4) again from positions a third of a cell off.
    _, default_line = line_sample(solver, Q, :rho)
    _, first_line = line_sample(solver, Q, :rho; index=(1, 1))
    @test default_line == first_line
    _, snapped = line_sample(solver, Q, :rho;
                             at=(gx(2, 7) + solver.h[2] / 3, gx(3, 4) - solver.h[3] / 3))
    _, indexed = line_sample(solver, Q, :rho; index=(7, 4))
    @test snapped == indexed

    # The plane average is a different quantity when a transverse dimension is
    # resolved, and the same one when both are collapsed.
    _, mean_line = line_profile(solver, Q, :rho; dim=1)
    @test maximum(abs, mean_line .- first_line) > 0.1
    solver1 = build_solver(
        Problem(domain=((0.0, 2.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(32, 1, 1), art=ArtificialProperties(enabled=false)))
    Q1 = allocate_state(solver1)
    initialize!(solver1, Q1, ic)
    _, sample1 = line_sample(solver1, Q1, :rho)
    _, mean1 = line_profile(solver1, Q1, :rho)
    @test maximum(abs, sample1 .- mean1) < 1e-14

    @test_throws ArgumentError line_sample(solver, Q, :rho; dim=4)
    @test_throws ArgumentError line_sample(solver, Q, :rho; index=(0, 1))
    @test_throws ArgumentError line_sample(solver, Q, :rho; dim=1, index=(1, 13))
    @test_throws ArgumentError line_sample(solver, Q, :rho; index=(1, 1), at=(0.0, 0.0))
end

@testset "diagnostics: quadrature, plane averages, mixing measures" begin
    # The quadrature is the load-bearing part: every mixing number is a ratio of
    # two of these integrals, so a wrong edge weight biases θ and W silently
    # rather than failing. Cartesian must be exact, and so must a constant over
    # a linear or quadratic Jacobian once the metric's share of each edge term
    # is removed (see the note in diagnostics.jl).
    solver = build_solver(
        Problem(domain=((0.0, 2.0), (0.0, 1.0), (0.0, 0.5)),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(24, 16, 12), art=ArtificialProperties(enabled=false)))
    @test domain_volume(solver) ≈ 1.0 atol = 1e-12          # 2.0 × 1.0 × 0.5
    ones_f = CL.field(solver.decomp); fill!(ones_f, 1.0)
    @test volume_integral(solver, ones_f) ≈ 1.0 atol = 1e-12
    @test volume_average(solver, ones_f) ≈ 1.0 atol = 1e-12
    # ∫x dV over x ∈ [0,2] with unit transverse area = 1.0 (trapezoid is exact
    # for a linear integrand, as provided by the half edge weight).
    lin = CL.field(solver.decomp)
    fillf!(solver, lin, (x, y, z) -> x)
    @test volume_integral(solver, lin) ≈ 1.0 atol = 1e-12
    # A plane average of a function of x alone returns that function, and the
    # spacing profile sums to the extent.
    prof = plane_profile(solver, lin, 1)
    xs = profile_coordinate(solver, 1)
    @test length(prof) == 24
    @test maximum(abs, prof .- xs) < 1e-12
    @test sum(profile_spacing(solver, 1)) ≈ 2.0 atol = 1e-12

    # Cylindrical with the axis fold, spherical with the origin fold. The
    # midpoint rule at the axis and the trapezoid rule at the outer wall err by
    # h²/24 and h²/12 on ∫r dr; the axis node's weight, 11/12 of the
    # midpoint's, and the wall node's remove both, so the linear J = r and the
    # quadratic J = r² integrate exactly. The origin has no such term.
    rz(N) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=false)))
    cyl = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(64, 1, 12), art=ArtificialProperties(enabled=false)))
    ones_c = CL.field(cyl.decomp); fill!(ones_c, 1.0)
    @test volume_integral(cyl, ones_c) ≈ 0.5 atol = 1e-14  # ∫r dr dθ dz, θ collapsed
    @test CL._edge_factor(cyl, 1, 1, padded_index(cyl, 1, 1, 1)) ≈ 11 / 12
    sph = build_solver(
        Problem(domain=((0.0, 0.0 + 1.0), (π / 2, π / 2 + 1.0), (0.0, 0.0 + 1.0)),
                metric=SphericalMetric(),
                bcs=((OriginBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(48, 1, 1), art=ArtificialProperties(enabled=false)))
    @test CL._edge_factor(sph, 1, 1, padded_index(sph, 1, 1, 1)) == 1.0
    @test domain_volume(sph) ≈ sin(xcoord(sph, 2, 1)) / 3 atol = 1e-14
    # A field even across the axis whose derivative vanishes at the wall
    # converges at fourth order: ∫ r (2 + cos πr) dr = 1 − 2/π².
    function rz_error(N)
        s = rz(N)
        f = fillf!(s, CL.field(s.decomp), (r, θ, z) -> 2 + cos(π * r))
        return abs(volume_integral(s, f) - (1 - 2 / π^2))
    end
    @test rz_error(32) / rz_error(64) > 12

    # Mixing measures against states whose answers are definitional.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 1.0, 1.4)])
    mk(ic) = begin
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 0.2), (0.0, 0.2)), bcs=per3, eos=eos,
                    ic=nothing),
            Numerics(n_global=(64, 12, 12), art=ArtificialProperties(enabled=false)))
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        CL.exchange_state!(Q, s.decomp); CL.primitives!(s, Q)
        s, Q
    end
    # The pair W and θ exists precisely to separate stirring from mixing, and
    # these three states are the demonstration: the first two have IDENTICAL
    # mix width and opposite θ.
    #
    # (a) uniformly mixed everywhere: W is the full extent, θ = 1.
    s1, Q1 = mk((x, y, z) -> Prim(Y=(0.5, 0.5), p=1.0, rho=1.0))
    @test mix_width(s1, Q1) ≈ 1.0 atol = 1e-12
    @test molecular_mixing(s1, Q1) ≈ 1.0 atol = 1e-12
    # (b) stirred but not mixed: each plane is half pure a and half pure b, so
    # ⟨Y_a⟩⟨Y_b⟩ is unchanged from (a) but ⟨Y_a Y_b⟩ vanishes pointwise.
    s2, Q2 = mk((x, y, z) -> Prim(Y=(y < 0.1 ? 1.0 : 0.0, y < 0.1 ? 0.0 : 1.0),
                                  p=1.0, rho=1.0))
    @test mix_width(s2, Q2) ≈ 1.0 atol = 1e-12
    @test molecular_mixing(s2, Q2) < 1e-12
    # (c) an x-only interface: nothing varies within a plane, so θ is 1 by
    # construction and W collapses onto the interface.
    s3, Q3 = mk((x, y, z) -> begin
        θ = tanh_blend(x, 0.5, 1 / 64)
        Prim(Y=(1 - θ, θ), p=1.0, rho=1.0)
    end)
    @test mix_width(s3, Q3) < 0.05
    @test molecular_mixing(s3, Q3) ≈ 1.0 atol = 1e-12
    # The PDF of a segregated field piles up at 0 and 1 and integrates to 1.
    centers, pdf = species_pdf(s2, 1; nbins=20)
    @test sum(pdf) * (centers[2] - centers[1]) ≈ 1.0 atol = 1e-10
    @test pdf[1] + pdf[end] > 0.99 * sum(pdf)
    _, pdf1 = species_pdf(s1, 1; nbins=20)
    @test pdf1[11] > 0.99 * sum(pdf1)                # all mass in the Y=0.5 bin

    # TKE removes the plane mean, so a uniform stream carries none; a
    # transverse-varying velocity does.
    s4, Q4 = mk((x, y, z) -> Prim(Y=(1.0, 0.0), u=(0.7, 0, 0), p=1.0, rho=1.0))
    @test turbulent_kinetic_energy(s4, Q4) < 1e-20
    @test maximum(abs, tke_profile(s4, Q4)) < 1e-20
    s5, Q5 = mk((x, y, z) -> Prim(Y=(1.0, 0.0), u=(0.7, 0.3sin(2π * y / 0.2), 0),
                                  p=1.0, rho=1.0))
    @test turbulent_kinetic_energy(s5, Q5) > 1e-3
    # Dissipation is a sink: zero for a uniform state, positive under shear.
    @test abs(dissipation_rate(s4, Q4)) < 1e-20
    s6, Q6 = mk((x, y, z) -> Prim(Y=(1.0, 0.0), u=(0.0, 0.3sin(2π * x), 0),
                                  p=1.0, rho=1.0))
    s6.transport = ConstantTransport(mu0=1e-2)
    @test dissipation_rate(s6, Q6) > 0
end

@testset "save_vtk writes a readable .pvtr/.vtr pair" begin
    solver = mkslv(n_global=(12, 12, 12))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    save_vtk(solver, Q, "test_vtk")
    files = filter(startswith("test_vtk"), readdir())
    @test any(endswith(".pvtr"), files)
    @test any(endswith(".vtr"), files)
    @test all(f -> filesize(f) > 0, files)
    txt = read(first(filter(endswith(".pvtr"), files)), String)
    @test occursin("PRectilinearGrid", txt)
    foreach(rm, files)
end

# Walk the appended-data blocks of a .vtr/.vts piece and return the Float32
# point-data arrays by name. Each block is a UInt64 byte count followed by its
# payload, so this also verifies that the offsets and lengths are consistent. A
# writer that miscounted desynchronizes here rather than producing a plausible
# file that only ParaView rejects.
function read_vtk_pointdata(path)
    raw = read(path)
    marker = Vector{UInt8}("<AppendedData encoding=\"raw\">\n_")
    i0 = findfirst(marker, raw)[end]
    header = String(copy(raw[1:i0]))
    decls = [(m.captures[1], parse(Int, m.captures[2])) for m in eachmatch(
        r"<DataArray type=\"Float32\" Name=\"([^\"]+)\" NumberOfComponents=\"(\d+)\"",
        header)]
    blocks = Vector{Vector{UInt8}}()
    pos = i0 + 1
    stop = length(raw) - length("\n</AppendedData>\n</VTKFile>\n")
    while pos <= stop
        n = only(reinterpret(UInt64, raw[pos:pos+7]))
        push!(blocks, raw[pos+8:pos+7+n])
        pos += 8 + n
    end
    @assert length(blocks) >= length(decls)
    geom = blocks[1:end-length(decls)]           # coordinates or points
    out = Dict{String,Vector{Float32}}()
    for (m, (name, nc)) in enumerate(decls)
        arr = collect(reinterpret(Float32, blocks[end-length(decls)+m]))
        out[name] = arr
    end
    return out, geom, decls
end

@testset "save_vtk: field selection and derived fields" begin
    dir = mktempdir()
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(32, 32, 12), art=ArtificialProperties(enabled=true)))
    Q = allocate_state(solver)

    # (a) A uniform stream, for which every derived field is zero by
    # definition. This detects a sign error or a transposed index in the curl.
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    derived = (:vorticity, :vorticity_magnitude, :qcriterion, :divergence,
               :schlieren, :mach)
    save_vtk(solver, Q, joinpath(dir, "uniform"); fields=derived)
    got, _, decls = read_vtk_pointdata(joinpath(dir, "uniform.r0000.vtr"))
    @test [n for (n, _) in decls] == ["vorticity", "vorticity_magnitude",
                                      "qcriterion", "divergence", "schlieren", "mach"]
    for name in ("vorticity", "vorticity_magnitude", "qcriterion", "divergence",
                 "schlieren")
        @test maximum(abs, got[name]) < 1e-5
    end
    @test all(≈(0.3 / sqrt(1.4)), got["mach"])        # |u|/c, γRT = 1.4 here

    # (b) A shear layer u₁ = sin(2πy), for which ω₃ = −∂u₁/∂x₂ = −2π cos(2πy)
    # and the divergence is zero. The interior scheme is C6, so the tolerance
    # is tight.
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(2π * y), 0, 0), p=1.0, rho=1.0))
    save_vtk(solver, Q, joinpath(dir, "shear"); fields=(:vorticity, :divergence))
    got, _, _ = read_vtk_pointdata(joinpath(dir, "shear.r0000.vtr"))
    ω = got["vorticity"]
    nx, ny = 32, 32
    werr = 0.0
    for j in 1:ny, i in 1:nx
        m = (j - 1) * nx + i                          # k = 1 plane, i fastest
        want = -2π * cos(2π * xcoord(solver, 2, j))
        werr = max(werr, abs(ω[3m] - want))           # third component
    end
    @test werr < 1e-4
    @test maximum(abs, got["divergence"]) < 1e-4

    # (c) The default set is unchanged from before field selection was added.
    save_vtk(solver, Q, joinpath(dir, "default"))
    _, _, decls = read_vtk_pointdata(joinpath(dir, "default.r0000.vtr"))
    @test [n for (n, _) in decls] == ["rho", "velocity", "p", "T_ion", "Y1"]
    @test [nc for (_, nc) in decls] == [1, 3, 1, 1, 1]

    # (d) The artificial coefficients are readable, which is the reason for
    # exposing them: they report the local action of the regularization.
    save_vtk(solver, Q, joinpath(dir, "art"); fields=(:sensor, :mu_art, :D_art))
    _, _, decls = read_vtk_pointdata(joinpath(dir, "art.r0000.vtr"))
    @test [n for (n, _) in decls] == ["sensor", "mu_art", "D_art1"]

    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        fields=(:not_a_field,))
    rm(dir; recursive=true)
end

@testset "save_vtk: strided subsampling" begin
    dir = mktempdir()
    N = (32, 16, 16)
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=N, art=ArtificialProperties(enabled=false)))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                             rho=1 + x + 2y + 4z))

    # Coordinates and payload for a given stride, read back from the piece.
    strided(stem, stride) = begin
        save_vtk(solver, Q, stem; fields=(:rho,), stride=stride)
        got, geom, _ = read_vtk_pointdata(stem * ".r0000.vtr")
        coords = [collect(reinterpret(Float64, g)) for g in geom]
        header = String(copy(read(stem * ".pvtr")))
        whole = match(r"WholeExtent=\"([^\"]+)\"", header).captures[1]
        (coords, got["rho"], whole)
    end

    # Stride 1 reproduces the unstrided grid, confirming that the general path
    # introduces no index shift.
    coords, rho1, whole = strided(joinpath(dir, "s1"), 1)
    @test length.(coords) == [32, 16, 16]
    @test whole == "0 31 0 15 0 15"
    for d in 1:3
        @test coords[d] ≈ [xcoord(solver, d, i) for i in 1:N[d]] atol = 1e-15
    end

    # Stride 2 gives half the points per dimension and extents in the coarse
    # index space. The coordinates are the ODD global stations 1, 3, 5, ...,
    # not a uniformly re-spaced grid.
    coords, rho2, whole = strided(joinpath(dir, "s2"), 2)
    @test length.(coords) == [16, 8, 8]
    @test whole == "0 15 0 7 0 7"
    for d in 1:3
        @test coords[d] ≈ [xcoord(solver, d, i) for i in 1:2:N[d]] atol = 1e-15
    end
    @test length(rho2) == 16 * 8 * 8

    # The payload holds the same values, sampled rather than averaged or
    # shifted.
    want = Float32[]
    for k in 1:2:16, j in 1:2:16, i in 1:2:32
        push!(want, Float32(solver.rho[padded_index(solver, i, j, k)]))
    end
    @test rho2 == want

    # Per-dimension strides are independent.
    coords, _, whole = strided(joinpath(dir, "s421"), (4, 2, 1))
    @test length.(coords) == [8, 8, 16]
    @test whole == "0 7 0 7 0 15"

    # An odd extent keeps the ceiling: 1, 3, ..., 15 out of 15 is 8 stations.
    odd = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(15, 16, 16), art=ArtificialProperties(enabled=false)))
    Qo = allocate_state(odd)
    initialize!(odd, Qo, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
    save_vtk(odd, Qo, joinpath(dir, "odd"); fields=(:rho,), stride=2)
    @test occursin("WholeExtent=\"0 7 0 7 0 7\"",
                   String(copy(read(joinpath(dir, "odd.pvtr")))))

    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad"); stride=0)
    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        stride=(2, 0, 1))
    rm(dir; recursive=true)
end

@testset "save_vtk: slicing" begin
    dir = mktempdir()
    N = (24, 16, 12)
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=N, art=ArtificialProperties(enabled=false)))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                             rho=1 + x + 100y + 10000z))
    refresh_primitives!(solver, Q)

    # A slice is one point thick in the sliced dimension and full in the others.
    for (d, g) in ((1, 7), (2, 5), (3, 9))
        stem = joinpath(dir, "s$(d)_$(g)")
        save_vtk(solver, Q, stem; fields=(:rho,), slice=(d, g))
        want = ntuple(t -> t == d ? 1 : N[t], 3)
        header = String(copy(read(stem * ".pvtr")))
        @test occursin("WholeExtent=\"0 $(want[1]-1) 0 $(want[2]-1) " *
                       "0 $(want[3]-1)\"", header)
        got, geom, _ = read_vtk_pointdata(stem * ".r0000.vtr")
        coords = [collect(reinterpret(Float64, b)) for b in geom]
        @test length.(coords) == [want...]
        # The retained station is the requested one, not the first.
        @test coords[d] ≈ [xcoord(solver, d, g)] atol = 1e-15
        @test length(got["rho"]) == prod(want)
        # And the values are the plane at g, not some other plane.
        idx = ntuple(t -> t == d ? (g:g) : (1:N[t]), 3)
        expect = Float32[solver.rho[padded_index(solver, i, j, k)]
                         for i in idx[1], j in idx[2], k in idx[3]][:]
        @test got["rho"] == expect
    end

    # A slice composes with a stride on the two dimensions still resolved.
    stem = joinpath(dir, "both")
    save_vtk(solver, Q, stem; fields=(:rho,), stride=2, slice=(3, 5))
    header = String(copy(read(stem * ".pvtr")))
    @test occursin("WholeExtent=\"0 11 0 7 0 0\"", header)
    _, geom, _ = read_vtk_pointdata(stem * ".r0000.vtr")
    @test length.([collect(reinterpret(Float64, b)) for b in geom]) == [12, 8, 1]

    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        slice=(4, 1))
    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        slice=(2, 0))
    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        slice=(2, 17))
    rm(dir; recursive=true)
end

@testset "save_vtk: a resolved angle writes a curvilinear grid" begin
    dir = mktempdir()
    # A cylindrical annulus with θ resolved. Written as a rectilinear grid it
    # would render unwrapped, as a box in (r, θ) rather than as an annulus.
    bcs = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
           (PeriodicBC(), PeriodicBC()))
    nr, nθ, nz = 12, 24, 10
    solver = build_solver(
        Problem(domain=((0.5, 0.5 + 1.0), (0.0, 0.0 + 2π), (0.0, 0.0 + 1.0)), bcs=bcs,
                metric=CylindricalMetric(), ic=nothing),
        Numerics(n_global=(nr, nθ, nz), art=ArtificialProperties(enabled=false)))
    Q = allocate_state(solver)
    # Solid-body swirl: purely azimuthal, constant magnitude.
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(0.0, 0.5, 0.0), p=1.0, rho=1.0))

    @test container_extension(solver) == ".pvts"
    save_vtk(solver, Q, joinpath(dir, "annulus"))
    @test isfile(joinpath(dir, "annulus.pvts"))
    @test isfile(joinpath(dir, "annulus.r0000.vts"))
    @test occursin("PStructuredGrid", read(joinpath(dir, "annulus.pvts"), String))

    got, geom, _ = read_vtk_pointdata(joinpath(dir, "annulus.r0000.vts"))
    pts = collect(reinterpret(Float64, only(geom)))
    @test length(pts) == 3 * nr * nθ * nz
    radii = [hypot(pts[3m-2], pts[3m-1]) for m in 1:(nr*nθ*nz)]
    @test minimum(radii) ≈ 0.5 rtol = 1e-12
    @test maximum(radii) ≈ 1.5 rtol = 1e-12

    # The velocity must be rotated into the frame in which the points are
    # written. Unrotated, (u_r, u_θ, u_z) = (0, 0.5, 0) would appear as a
    # uniform y-directed stream; rotated, it is tangent to every circle.
    vel = got["velocity"]
    radial = 0.0; magnitude = 0.0
    for m in 1:(nr*nθ*nz)
        x, y = pts[3m-2], pts[3m-1]
        ux, uy = Float64(vel[3m-2]), Float64(vel[3m-1])
        radial = max(radial, abs(ux * x + uy * y) / hypot(x, y))
        magnitude = max(magnitude, abs(hypot(ux, uy) - 0.5))
    end
    @test radial < 1e-6                    # no radial component: purely tangential
    @test magnitude < 1e-6                 # and the swirl speed is preserved

    # An axisymmetric (θ-collapsed) polar run remains rectilinear, since the
    # meridional half-plane is the computed domain and a rectangle represents
    # it correctly.
    axi = build_solver(
        Problem(domain=((0.5, 0.5 + 1.0), (0.0, 0.0 + 2π), (0.0, 0.0 + 1.0)),
                bcs=(bcs[1], (PeriodicBC(), PeriodicBC()), bcs[3]),
                metric=CylindricalMetric(), ic=nothing),
        Numerics(n_global=(nr, 1, nz), art=ArtificialProperties(enabled=false)))
    @test container_extension(axi) == ".pvtr"
    rm(dir; recursive=true)
end

@testset "FieldWriter: numbered frames and a .pvd carrying physical time" begin
    dir = mktempdir()
    solver = build_solver(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                  bcs=((SlipWallBC(), SlipWallBC()),
                                       (PeriodicBC(), PeriodicBC()),
                                       (PeriodicBC(), PeriodicBC())),
                                  ic=nothing),
                          Numerics(n_global=(16, 12, 12),
                                   art=ArtificialProperties(enabled=false), cfl=0.4))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                             p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
    writer = FieldWriter(joinpath(dir, "frames", "field"))
    run!(solver, Q; tfinal=0.03,
         callback=Callback(EveryTime(0.01), writer))

    # One frame per instant, the initial state first, numbered from zero, with
    # the directory created.
    @test writer.index == 4
    @test writer.times ≈ [0.0, 0.01, 0.02, 0.03] rtol = 1e-14
    for m in 0:3
        @test isfile(CL.frame_prefix(writer, m) * ".pvtr")
        @test filesize(CL.frame_prefix(writer, m) * ".pvtr") > 0
    end

    # The collection allows the sequence to animate against physical time, not
    # frame index, so its timesteps are the assertion.
    pvd = read(joinpath(dir, "frames", "field.pvd"), String)
    @test occursin("type=\"Collection\"", pvd)
    @test count("<DataSet", pvd) == 4
    stamps = [parse(Float64, m.captures[1])
              for m in eachmatch(r"timestep=\"([^\"]+)\"", pvd)]
    @test stamps ≈ [0.0, 0.01, 0.02, 0.03] rtol = 1e-14
    # Pieces are named relative to the .pvd's own directory rather than by
    # absolute path, so the collection remains readable on another machine.
    @test occursin("file=\"field_0000.pvtr\"", pvd)
    @test !occursin(dir, pvd)

    @test writer.wall_io > 0            # I/O cost is not in solver.wall_step
    rm(dir; recursive=true)
end
