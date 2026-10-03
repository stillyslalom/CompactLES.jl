# Every accepted combination of docs/src/reference/capabilities.md, each built
# at a small size and advanced three steps, with checkpoint round trips. The
# rejections and a smoke subset of these rows run in the serial suite
# (test/capability_tests.jl); this file takes about 80 s, which is out of
# proportion to a push job, so the weekly validation workflow runs it.
#
# Standalone: julia --project=. -O1 test/capability_matrix.jl

using Test
using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES.Regions: Box

include("capability_cases.jl")

@testset "capability matrix: accepted" begin
    C = Capabilities
    per, wall, axis = C.per, C.wall, C.axis
    stretched = (sine_cluster(0.0, 1.0, 0.5, 0.5), nothing, nothing)
    t0 = time()

    @testset "EOS × backend × precision" begin
        for (name, eos, ic) in C.EOSES, backend in (CompactLES.CPUBackend(), C.device()),
            precision in (Float64, Float32)
            ok = C.advances(C.problem(; eos, ic),
                            Numerics(; n_global=(32, 1, 1),
                                     execution=Execution(; backend, precision)))
            ok || @warn "capability matrix: $name on $backend at $precision did not advance"
            @test ok
        end
    end

    @testset "geometry × backend × precision" begin
        geometries = [
            ("stretched Cartesian", C.problem(bcs=(wall, per, per)),
             (; n_global=(32, 1, 1), stretch=stretched)),
            ("cylindrical axis, θ collapsed",
             C.problem(bcs=axis, metric=CylindricalMetric(), ic=C.ic_radial),
             (; n_global=(24, 1, 1))),
            ("cylindrical axis, θ resolved",
             C.problem(bcs=axis, domain=C.cyl_domain, metric=CylindricalMetric(),
                       ic=C.ic_radial),
             (; n_global=(16, 16, 1))),
            ("spherical origin",
             C.problem(bcs=((OriginBC(), SlipWallBC()), per, per),
                       domain=((0.0, 1.0), (π / 2 - 0.5, π / 2 + 0.5), (0.0, 1.0)),
                       metric=SphericalMetric(), ic=C.ic_radial),
             (; n_global=(24, 1, 1))),
            ("spherical poles",
             C.problem(bcs=(wall, (PoleBC(), PoleBC()), per),
                       domain=((0.5, 1.0), (0.0, π), (0.0, 2π)),
                       metric=SphericalMetric(), ic=C.ic_shell),
             (; n_global=(16, 16, 1))),
            ("Cartesian symmetry plane",
             C.problem(bcs=((SymmetryPlaneBC(), SlipWallBC()), per, per)),
             (; n_global=(32, 1, 1))),
            ("cylindrical z symmetry plane",
             C.problem(bcs=(axis[1], per, (SymmetryPlaneBC(), SlipWallBC())),
                       metric=CylindricalMetric(),
                       ic=(r, θ, z) -> Prim(p=1.0 + 0.1exp(-20(r^2 + z^2)), rho=1.0)),
             (; n_global=(16, 1, 16))),
        ]
        for (name, prob, kw) in geometries, (backend, precision) in C.PAIRS
            ok = C.advances(prob, Numerics(; execution=Execution(; backend, precision),
                                           kw...))
            ok || @warn "capability matrix: $name on $backend at $precision did not advance"
            @test ok
        end
        # Azimuthal mode truncation: host only, with or without the axis.
        for prob in (C.problem(bcs=axis, domain=C.cyl_domain, metric=CylindricalMetric(),
                               ic=C.ic_radial),
                     C.problem(bcs=(wall, per, per),
                               domain=((0.5, 1.0), (0.0, 2π), (0.0, 1.0)),
                               metric=CylindricalMetric(), ic=C.ic_shell))
            @test C.advances(prob, Numerics(n_global=(16, 32, 1), polar_truncation=2.0))
        end
        # The positivity limiter: host, one patch, Cartesian, closed and periodic
        # lines, one to three dimensions, one or two species.
        for (eos, ic) in (C.EOSES[1][2:3], C.EOSES[2][2:3]), n in ((64, 1, 1), (64, 16, 1),
                                                                 (56, 12, 12))
            @test C.advances(C.problem(bcs=(C.wall, C.per, C.per), eos=eos, ic=ic),
                             Numerics(n_global=n, positivity_limiter=true))
        end
        # The radial grids folded at the cylindrical axis and the spherical
        # origin, and the r-z plane, periodic in z or with a plane at z = 0.
        for (prob, n) in ((C.problem(bcs=C.axis, domain=C.cyl_domain,
                                     metric=CylindricalMetric(), ic=C.ic_radial), (64, 1, 1)),
                          (C.problem(bcs=((OriginBC(), SlipWallBC()), C.per, C.per),
                                     domain=((0.0, 1.0), (π / 2, π / 2 + 1), (0.0, 1.0)),
                                     metric=SphericalMetric(), ic=C.ic_radial), (64, 1, 1)),
                          (C.problem(bcs=C.axis, domain=C.cyl_domain,
                                     metric=CylindricalMetric(), ic=C.ic_radial), (48, 1, 16)),
                          (C.problem(bcs=((AxisBC(), SlipWallBC()), C.per,
                                          (SymmetryPlaneBC(), SlipWallBC())),
                                     domain=C.cyl_domain, metric=CylindricalMetric(),
                                     ic=C.ic_radial), (48, 1, 48)))
            @test C.advances(prob, Numerics(n_global=n, positivity_limiter=true))
        end
    end

    @testset "layouts × backend × precision" begin
        feature = (x, y, z, t) -> abs(x - 0.5) < 0.1
        layouts = [
            ("patch_grid", (; patch_grid=(2, 1, 1))),
            ("static nested levels",
             (; amr=AMR(initial=[Box((0.25, 0, 0), (0.75, 1, 1)),
                                 Box((0.4, 0, 0), (0.6, 1, 1))]))),
            ("regridded box", (; amr=AMR(initial=feature, regrid_interval=1))),
            ("regridded tiles", (; amr=AMR(initial=feature, regrid_interval=1, tile=4))),
            ("subcycled levels",
             (; amr=AMR(initial=Box((0.3, 0, 0), (0.7, 1, 1)), subcycle=true))),
        ]
        for (name, kw) in layouts, (backend, precision) in C.PAIRS
            execution = Execution(; backend, precision,
                                  patch_grid=get(kw, :patch_grid, (1, 1, 1)))
            ok = C.advances(C.problem(), Numerics(; n_global=(48, 1, 1), execution,
                                                  amr=get(kw, :amr, nothing)))
            ok || @warn "capability matrix: $name on $backend at $precision did not advance"
            @test ok
        end
        # More than two regridded levels: tiles on the host backend.
        @test C.advances(C.problem(),
                         Numerics(n_global=(48, 1, 1),
                                  amr=AMR(initial=feature, regrid_interval=1, tile=4,
                                          max_levels=3)))
        # Nested BlockRegions regrid with tiles.
        @test C.advances(C.problem(),
                         Numerics(n_global=(48, 1, 1),
                                  amr=AMR(initial=[BlockRegion((16, 0, 0), (16, 1, 1)),
                                                   BlockRegion((60, 0, 0), (20, 1, 1))],
                                          regrid_interval=1, tile=4)))
        # A level reaching a symmetry plane, on the host backend at either
        # precision, and a regridded level off it.
        plane = C.problem(bcs=((SymmetryPlaneBC(), SlipWallBC()), per, per))
        for precision in (Float64, Float32)
            @test C.advances(plane, Numerics(n_global=(48, 1, 1),
                                             execution=Execution(; precision),
                                             amr=AMR(initial=BlockRegion((0, 0, 0),
                                                                         (12, 1, 1)))))
        end
        @test C.advances(plane, Numerics(n_global=(48, 1, 1),
                                         amr=AMR(initial=feature, regrid_interval=1)))
        # Levels placed on a wall by tags, a box and tiles, on either backend,
        # and a static shape's level on a symmetry plane and on the axis.
        walled = C.problem(bcs=(wall, per, per))
        at_wall = (x, y, z, t) -> x > 0.85
        for tile in (0, 8), (backend, precision) in C.PAIRS
            ok = C.advances(walled, Numerics(n_global=(49, 1, 1),
                                             execution=Execution(; backend, precision),
                                             amr=AMR(initial=at_wall, regrid_interval=1,
                                                     tile=tile, level_boundaries=true)))
            ok || @warn "capability matrix: placed level (tile $tile) on $backend at " *
                        "$precision did not advance"
            @test ok
        end
        @test C.advances(plane, Numerics(n_global=(48, 1, 1),
                                         amr=AMR(initial=Box((0.0, 0, 0), (0.2, 1, 1)),
                                                 level_boundaries=true)))
        @test C.advances(C.problem(bcs=axis, metric=CylindricalMetric(), ic=C.ic_radial),
                         Numerics(n_global=(48, 1, 1),
                                  amr=AMR(initial=Box((0.0, 0, 0), (0.2, 1, 1)),
                                          level_boundaries=true)))
        # A feature at a symmetry plane is refined up to the plane on the
        # device backend and under the :filter restriction too.
        near_plane = (x, y, z, t) -> x < 0.15
        for (backend, precision) in C.PAIRS
            backend isa DeviceBackend || continue
            @test C.advances(plane, Numerics(n_global=(49, 1, 1),
                                             execution=Execution(; backend, precision),
                                             amr=AMR(initial=near_plane,
                                                     regrid_interval=1)))
            @test C.advances(plane, Numerics(n_global=(49, 1, 1),
                                             execution=Execution(; backend, precision),
                                             amr=AMR(initial=near_plane, regrid_interval=1,
                                                     tile=8)))
        end
        @test C.advances(plane, Numerics(n_global=(49, 1, 1),
                                         amr=AMR(initial=near_plane, regrid_interval=1,
                                                 level_restriction=:filter)))
        # Tags place a regridded box and tiles on a symmetry plane, nested
        # shapes reach it, and three regridded levels reach the corner of the
        # axis and a symmetry plane at z = 0.
        at_plane = (x, y, z, t) -> x < 0.15
        for tile in (0, 8)
            @test C.advances(plane, Numerics(n_global=(49, 1, 1),
                                             amr=AMR(initial=at_plane, regrid_interval=1,
                                                     tile=tile, level_boundaries=true)))
        end
        @test C.advances(plane, Numerics(n_global=(48, 1, 1),
                                         amr=AMR(initial=[Box((0.0, 0, 0), (0.3, 1, 1)),
                                                          Box((0.0, 0, 0), (0.1, 1, 1))],
                                                 level_boundaries=true)))
        @test C.advances(C.problem(bcs=(axis[1], per, (SymmetryPlaneBC(), SlipWallBC())),
                                   metric=CylindricalMetric(),
                                   ic=(r, θ, z) -> Prim(p=1.0 + 0.1exp(-20(r^2 + z^2)),
                                                        rho=1.0)),
                         Numerics(n_global=(32, 1, 32),
                                  amr=AMR(initial=(r, θ, z, t) -> r^2 + z^2 < 0.04,
                                          regrid_interval=1, tile=4, max_levels=3,
                                          level_boundaries=true)))
        # A level across a periodic seam: an explicit box, and tags placing a
        # box and tiles there, on either backend.
        near_seam = (x, y, z, t) -> x < 0.06 || x > 0.94
        for (backend, precision) in C.PAIRS
            execution = Execution(; backend, precision)
            ok = C.advances(C.problem(), Numerics(; n_global=(48, 1, 1), execution,
                                                  amr=AMR(initial=BlockRegion((44, 0, 0),
                                                                              (9, 1, 1)))))
            for tile in (0, 8)
                ok &= C.advances(C.problem(),
                                 Numerics(; n_global=(48, 1, 1), execution,
                                          amr=AMR(initial=near_seam, regrid_interval=1,
                                                  tile=tile, level_boundaries=true)))
            end
            ok || @warn "capability matrix: a level across the seam on $backend at " *
                        "$precision did not advance"
            @test ok
        end
        # Every EOS on a patched and on a refined layout.
        for (name, eos, ic) in C.EOSES[2:end],
            kw in ((; execution=Execution(patch_grid=(2, 1, 1))),
                   (; amr=AMR(initial=feature, regrid_interval=1)))
            @test C.advances(C.problem(; eos, ic), Numerics(; n_global=(48, 1, 1), kw...))
        end
        # Slabs on a cylindrical annulus and along a uniform dimension of a
        # stretched grid, which take the closure rows; the ghost-flux
        # interface, the default, on slabs and on a viscous level.
        annulus = C.problem(bcs=(wall, per, per), metric=CylindricalMetric(),
                            domain=((0.5, 1.5), (0.0, 1.0), (0.0, 1.0)),
                            ic=(r, θ, z) -> Prim(p=1.0 + 0.1exp(-20(r - 1)^2), rho=1.0))
        for backend in (CompactLES.CPUBackend(), C.device())
            @test C.advances(annulus, Numerics(n_global=(48, 1, 1),
                                               patch_interfaces=:closure,
                                               execution=Execution(patch_grid=(2, 1, 1),
                                                                   backend=backend)))
        end
        @test C.advances(C.problem(bcs=(wall, per, per)),
                         Numerics(n_global=(16, 48, 1), patch_interfaces=:closure,
                                  execution=Execution(patch_grid=(1, 2, 1)),
                                  stretch=stretched))
        @test C.advances(C.problem(), Numerics(n_global=(48, 1, 1),
                                               execution=Execution(patch_grid=(2, 1, 1))))
        # Axisymmetric (θ-collapsed) layouts: slabs on the annulus under the
        # default ghost fluxes; a level on the annulus and on an axis root,
        # held off the axis, at both pairs; a viscous tiled regrid in r-z.
        @test C.advances(annulus, Numerics(n_global=(48, 1, 1),
                                           execution=Execution(patch_grid=(2, 1, 1))))
        axis_rz = C.problem(bcs=axis, metric=CylindricalMetric(), ic=C.ic_radial)
        for (backend, precision) in C.PAIRS
            execution = Execution(; backend, precision)
            @test C.advances(annulus, Numerics(; n_global=(48, 1, 1), execution,
                                               amr=AMR(initial=Box((0.9, 0, 0),
                                                                   (1.1, 1, 1)))))
            @test C.advances(axis_rz, Numerics(; n_global=(48, 1, 1), execution,
                                               amr=AMR(initial=Box((0.4, 0, 0),
                                                                   (0.6, 1, 1)),
                                                       subcycle=true)))
        end
        @test C.advances(C.problem(bcs=(axis[1], per, wall), metric=CylindricalMetric(),
                                   transport=ConstantTransport(mu0=1e-3),
                                   ic=(r, θ, z) -> Prim(p=1.0 + 0.1exp(-20((r - 0.5)^2 +
                                                                          (z - 0.5)^2)),
                                                        rho=1.0)),
                         Numerics(n_global=(32, 1, 32),
                                  amr=AMR(initial=(r, θ, z, t) -> abs(r - 0.5) < 0.1 &&
                                                                  abs(z - 0.5) < 0.1,
                                          regrid_interval=1, tile=4)))
        # A level reaching the axis, on the host backend at either precision,
        # and a viscous tiled level at the corner of the axis and a symmetry
        # plane at z = 0, the capsule's layout.
        for precision in (Float64, Float32)
            @test C.advances(axis_rz, Numerics(n_global=(48, 1, 1),
                                               execution=Execution(; precision),
                                               amr=AMR(initial=BlockRegion((0, 0, 0),
                                                                           (12, 1, 1)))))
        end
        @test C.advances(C.problem(bcs=(axis[1], per, (SymmetryPlaneBC(), SlipWallBC())),
                                   metric=CylindricalMetric(),
                                   transport=ConstantTransport(mu0=1e-3),
                                   ic=(r, θ, z) -> Prim(p=1.0 + 0.1exp(-20(r^2 + z^2)),
                                                        rho=1.0)),
                         Numerics(n_global=(32, 1, 32),
                                  amr=AMR(initial=BlockRegion((0, 0, 0), (13, 1, 13)),
                                          tile=4)))
        @test C.advances(C.problem(transport=ConstantTransport(mu0=1e-3)),
                         Numerics(n_global=(48, 1, 1),
                                  amr=AMR(initial=Box((0.3, 0, 0), (0.7, 1, 1)))))
    end

    @testset "checkpoints" begin
        dir = mktempdir()
        restarts(prob, num; reload=num) = begin
            solver, Q = setup(prob, num)
            run!(solver, Q; tfinal=1.0, nmax=2)
            save_checkpoint(solver, Q, joinpath(dir, "c"))
            solver2, Q2 = setup(prob, reload)
            load_checkpoint!(solver2, Q2, joinpath(dir, "c"))
            run!(solver2, Q2; tfinal=1.0, nmax=4)
            solver2.step == 4
        end
        n = (48, 1, 1)
        @test restarts(C.problem(), Numerics(n_global=n,
                                             execution=Execution(backend=C.device())))
        @test restarts(C.problem(eos=C.EOSES[4][2], ic=C.ic_air), Numerics(n_global=n))
        @test restarts(C.problem(), Numerics(n_global=n,
            amr=AMR(initial=(x, y, z, t) -> abs(x - 0.5) < 0.1, regrid_interval=1,
                    tile=4)))
        @test restarts(C.problem(), Numerics(n_global=n,
                                             execution=Execution(backend=C.device()),
                                             amr=AMR(initial=Box((0.3, 0, 0), (0.7, 1, 1)))))
    end
    println("capability matrix, accepted rows: $(round(time() - t0; digits=1)) s")
end
