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
                            Numerics(; n_global=(32, 1, 1), backend, precision))
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
            ok = C.advances(prob, Numerics(; backend, precision, kw...))
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
            ok = C.advances(C.problem(), Numerics(; n_global=(48, 1, 1), backend,
                                                  precision, kw...))
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
        # Every EOS on a patched and on a refined layout.
        for (name, eos, ic) in C.EOSES[2:end],
            kw in ((; patch_grid=(2, 1, 1)), (; amr=AMR(initial=feature, regrid_interval=1)))
            @test C.advances(C.problem(; eos, ic), Numerics(; n_global=(48, 1, 1), kw...))
        end
        # Slabs on a cylindrical annulus and along a uniform dimension of a
        # stretched grid; the ghost-flux interface on slabs and on a level.
        annulus = C.problem(bcs=(wall, per, per), metric=CylindricalMetric(),
                            domain=((0.5, 1.5), (0.0, 1.0), (0.0, 1.0)),
                            ic=(r, θ, z) -> Prim(p=1.0 + 0.1exp(-20(r - 1)^2), rho=1.0))
        for backend in (CompactLES.CPUBackend(), C.device())
            @test C.advances(annulus, Numerics(n_global=(48, 1, 1), patch_grid=(2, 1, 1),
                                               backend=backend))
        end
        @test C.advances(C.problem(bcs=(wall, per, per)),
                         Numerics(n_global=(16, 48, 1), patch_grid=(1, 2, 1),
                                  stretch=stretched))
        @test C.advances(C.problem(), Numerics(n_global=(48, 1, 1), patch_grid=(2, 1, 1),
                                               interface_flux=:ghost))
        @test C.advances(C.problem(transport=ConstantTransport(mu0=1e-3)),
                         Numerics(n_global=(48, 1, 1), interface_flux=:ghost,
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
        @test restarts(C.problem(), Numerics(n_global=n, backend=C.device()))
        @test restarts(C.problem(eos=C.EOSES[4][2], ic=C.ic_air), Numerics(n_global=n))
        @test restarts(C.problem(), Numerics(n_global=n,
            amr=AMR(initial=(x, y, z, t) -> abs(x - 0.5) < 0.1, regrid_interval=1,
                    tile=4)))
        @test restarts(C.problem(), Numerics(n_global=n, backend=C.device(),
                                             amr=AMR(initial=Box((0.3, 0, 0), (0.7, 1, 1)))))
    end
    println("capability matrix, accepted rows: $(round(time() - t0; digits=1)) s")
end
