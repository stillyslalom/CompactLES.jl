# The combinations of docs/src/reference/capabilities.md, held against the
# setup checks. Each rejected combination must raise the error the page
# quotes, and a smoke subset of the accepted rows, one host and one device
# row, is built at a small size and advanced three steps. A setup check
# changed without the page fails here. Every accepted row, with the
# checkpoint round trips, is in test/capability_matrix.jl, which the weekly
# validation workflow runs.
#
# Standalone: julia --project=. -O1 test/capability_tests.jl
# The serial suite includes it.

using Test
using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES.Regions: Box

include("capability_cases.jl")

@testset "capability matrix" begin
    C = Capabilities
    per, wall, axis = C.per, C.wall, C.axis
    stretched = (sine_cluster(0.0, 1.0, 0.5, 0.5), nothing, nothing)
    t0 = time()

    @testset "accepted: smoke rows" begin
        name, eos, ic = C.EOSES[1]
        for (backend, precision) in C.PAIRS
            ok = C.advances(C.problem(; eos, ic),
                            Numerics(; n_global=(32, 1, 1),
                                     execution=Execution(; backend, precision)))
            ok || @warn "capability matrix: $name on $backend at $precision did not advance"
            @test ok
        end
    end

    @testset "rejected at setup" begin
        gas = C.problem()
        cyl = C.problem(bcs=axis, domain=C.cyl_domain, metric=CylindricalMetric(),
                        ic=C.ic_radial)
        sph = C.problem(bcs=(wall, (PoleBC(), PoleBC()), per),
                        domain=((0.5, 1.0), (0.0, π), (0.0, 2π)),
                        metric=SphericalMetric(), ic=C.ic_shell)
        n1 = (48, 1, 1)
        box = AMR(initial=Box((0.3, 0, 0), (0.7, 1, 1)))
        feature = (x, y, z, t) -> abs(x - 0.5) < 0.1
        rejects(msg, prob, num) = @test_throws msg setup(prob, num)

        rejects("AxisBC requires CylindricalMetric",
                C.problem(bcs=axis), Numerics(n_global=(24, 1, 1)))
        rejects("OriginBC requires SphericalMetric",
                C.problem(bcs=((OriginBC(), SlipWallBC()), per, per)),
                Numerics(n_global=(24, 1, 1)))
        rejects("PoleBC must be applied at both ends of θ",
                C.problem(bcs=(wall, (PoleBC(), SlipWallBC()), per),
                          domain=((0.5, 1.0), (0.0, π), (0.0, 1.0)),
                          metric=SphericalMetric(), ic=C.ic_shell),
                Numerics(n_global=(16, 16, 1)))
        rejects("CylindricalMetric: radial node 1 lies on the axis r = 0",
                C.problem(bcs=(wall, per, per), domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)),
                          metric=CylindricalMetric(), ic=C.ic_shell),
                Numerics(n_global=(24, 1, 1)))
        rejects("SphericalMetric: radial node 1 lies on the origin r = 0",
                C.problem(bcs=(wall, per, per),
                          domain=((0.0, 1.0), (π / 2 - 0.5, π / 2 + 0.5), (0.0, 1.0)),
                          metric=SphericalMetric(), ic=C.ic_shell),
                Numerics(n_global=(24, 1, 1)))
        rejects("SphericalMetric: θ node 1 lies on a pole",
                C.problem(bcs=(wall, wall, per), domain=((0.5, 1.0), (0.0, π), (0.0, 1.0)),
                          metric=SphericalMetric(), ic=C.ic_shell),
                Numerics(n_global=(16, 16, 1)))
        rejects("SphericalMetric: θ node 16 lies on a pole",
                C.problem(bcs=(wall, wall, per), domain=((0.5, 1.0), (0.1, π), (0.0, 1.0)),
                          metric=SphericalMetric(), ic=C.ic_shell),
                Numerics(n_global=(16, 16, 1), execution=Execution(precision=Float32)))
        rejects("SphericalMetric: the collapsed θ node lies on a pole",
                C.problem(bcs=((OriginBC(), SlipWallBC()), per, per),
                          domain=((0.0, 1.0), (0.0, π), (0.0, 1.0)),
                          metric=SphericalMetric(), ic=C.ic_radial),
                Numerics(n_global=(24, 1, 1)))
        rejects("SymmetryPlaneBC on dimension 1 requires CartesianMetric or the z " *
                "dimension of CylindricalMetric",
                C.problem(bcs=((SymmetryPlaneBC(), SlipWallBC()), per, per),
                          domain=((0.5, 1.0), (0.0, 1.0), (0.0, 1.0)),
                          metric=CylindricalMetric(), ic=C.ic_shell),
                Numerics(n_global=(24, 1, 1)))
        rejects("the LODI formulation requires a face whose normal metric scale " *
                "factor is one",
                C.problem(bcs=(wall, (SlipWallBC(), NSCBCOutflowBC(pinf=1.0)), per),
                          domain=((0.5, 1.0), (0.0, 1.0), (0.0, 1.0)),
                          metric=CylindricalMetric(), ic=C.ic_shell),
                Numerics(n_global=(16, 16, 1)))
        rejects("folded dimensions cannot be stretched",
                C.problem(bcs=((SymmetryPlaneBC(), SlipWallBC()), per, per)),
                Numerics(n_global=(32, 1, 1), stretch=stretched))
        rejects("stretched dimensions must be non-periodic",
                gas, Numerics(n_global=(32, 1, 1), stretch=stretched))
        rejects("patch decomposition across a coordinate fold is not supported",
                cyl, Numerics(n_global=(16, 48, 1), execution=Execution(patch_grid=(1, 2, 1))))
        rejects("patch decomposition across a SymmetryPlaneBC is not supported",
                C.problem(bcs=(per, per, (SymmetryPlaneBC(), SlipWallBC()))),
                Numerics(n_global=(48, 1, 16), execution=Execution(patch_grid=(2, 1, 1))))
        rejects("patch interfaces carry closure variants for a tridiagonal filter only",
                gas, Numerics(n_global=n1, filter=pyranda_filter(),
                              execution=Execution(patch_grid=(2, 1, 1))))
        rejects("patch interfaces support the :delta4 and :species_d8 detectors only",
                gas, Numerics(n_global=n1, execution=Execution(patch_grid=(2, 1, 1)),
                              art=ArtificialProperties(detector=:d8)))
        rejects("an explicit process grid cannot combine with patch_grid",
                gas, Numerics(n_global=n1,
                              execution=Execution(patch_grid=(2, 1, 1), dims=(1, 1, 1))))
        rejects("the patched dimension cannot be stretched",
                C.problem(bcs=(wall, per, per)),
                Numerics(n_global=n1, execution=Execution(patch_grid=(2, 1, 1)),
                         stretch=stretched))
        rejects("AMR: cannot be combined with a patch_grid",
                gas, Numerics(n_global=n1, execution=Execution(patch_grid=(2, 1, 1)), amr=box))
        rejects("AMR: requires CartesianMetric or CylindricalMetric with θ collapsed",
                cyl, Numerics(n_global=(48, 16, 1), amr=box))
        rejects("AMR: requires CartesianMetric or CylindricalMetric with θ collapsed",
                sph, Numerics(n_global=(16, 16, 1), amr=box))
        switched = SwitchableBC(SlipWallBC(), NSCBCOutflowBC(pinf=1.0); at=1.0)
        rejects("a refined level cannot carry",
                C.problem(bcs=((SlipWallBC(), switched), per, per)),
                Numerics(n_global=n1, amr=AMR(initial=BlockRegion((38, 0, 0), (10, 1, 1)))))
        rejects("AMR: requires a uniform grid",
                C.problem(bcs=(wall, per, per)), Numerics(n_global=n1, stretch=stretched,
                                                          amr=box))
        # The symmetry plane and the r-z axis are refused alike; each row
        # takes the fold the other rows do not.
        plane = C.problem(bcs=((SymmetryPlaneBC(), SlipWallBC()), per, per))
        axis_rz = C.problem(bcs=axis, metric=CylindricalMetric(), ic=C.ic_radial)
        at_plane = BlockRegion((0, 0, 0), (12, 1, 1))
        rejects("a refined level reaching a SymmetryPlaneBC or an AxisBC runs on the " *
                "host backend only",
                plane, Numerics(n_global=n1, execution=Execution(backend=C.device()),
                                amr=AMR(initial=at_plane)))
        rejects("a refined level reaching a SymmetryPlaneBC or an AxisBC takes " *
                "level_restriction = :inject", axis_rz,
                Numerics(n_global=n1, amr=AMR(initial=at_plane, level_restriction=:filter)))
        # A second level reaching the axis starts on its parent's first node,
        # offset -1; one at offset 0 stops short of it inside the margin.
        rejects("offset -1 reaches the AxisBC of dimension 1",
                axis_rz, Numerics(n_global=n1, amr=AMR(initial=[at_plane,
                                                                BlockRegion((0, 0, 0),
                                                                            (12, 1, 1))])))
        rejects("AMR: regridding moves one refined level", gas,
                Numerics(n_global=n1, amr=AMR(initial=[Box((0.25, 0, 0), (0.75, 1, 1)),
                                                       Box((0.4, 0, 0), (0.6, 1, 1))],
                                              regrid_interval=1)))
        rejects("requires tile > 0 and regridding", gas,
                Numerics(n_global=n1, amr=AMR(initial=feature, max_levels=3)))
        rejects("regridding more than one refined level runs on the host backend only",
                gas, Numerics(n_global=n1, execution=Execution(backend=C.device()),
                              amr=AMR(initial=feature, regrid_interval=1, tile=4,
                                      max_levels=3)))
        rejects("level_restriction = :filter is host-only", gas,
                Numerics(n_global=n1, execution=Execution(backend=C.device()),
                         amr=AMR(initial=Box((0.3, 0, 0), (0.7, 1, 1)),
                                 level_restriction=:filter)))
        rejects("rebalance repartitions a tiled level at the regrid cadence", gas,
                Numerics(n_global=n1, amr=AMR(initial=feature, regrid_interval=1,
                                              rebalance=1.5)))
        rejects("interface_flux = :ghost (the default) requires an unstretched " *
                "CartesianMetric, or CylindricalMetric with θ collapsed, at a patch " *
                "or level interface; pass interface_flux = :closure",
                C.problem(bcs=(wall, per, per), metric=CylindricalMetric(),
                          domain=((0.5, 1.5), (0.0, 2π), (0.0, 1.0)), ic=C.ic_shell),
                Numerics(n_global=(48, 16, 1), execution=Execution(patch_grid=(2, 1, 1))))
        rejects("interface_flux = :ghost (the default) reads the gradient plans' " *
                "interface rows, which exist under interface_rhs = :extended only; " *
                "pass interface_flux = :closure",
                gas, Numerics(n_global=n1, patch_interfaces=PatchInterfaces(rhs=:onesided),
                              execution=Execution(patch_grid=(2, 1, 1))))
        rejects(["interface_flux = :ghost (the default) with molecular transport at a " *
                 "refined level supports IdealMixture, Nasa9Mixture and StiffenedGas",
                 "use interface_flux = :closure for this EOS"],
                C.problem(eos=C.EOSES[5][2], transport=ConstantTransport(mu0=1e-3)),
                Numerics(n_global=n1, amr=box))
        rejects("polar_truncation applies to CylindricalMetric", sph,
                Numerics(n_global=(16, 16, 1), polar_truncation=2.0))
        rejects("polar_truncation requires θ resolved and periodic over 2π",
                C.problem(bcs=axis, metric=CylindricalMetric(), ic=C.ic_radial),
                Numerics(n_global=(24, 1, 1), polar_truncation=2.0))
        rejects("polar_truncation requires an unstretched radial dimension",
                C.problem(bcs=(wall, per, per), domain=((0.5, 1.5), (0.0, 2π), (0.0, 1.0)),
                          metric=CylindricalMetric(), ic=C.ic_shell),
                Numerics(n_global=(16, 32, 1), polar_truncation=2.0,
                         stretch=(sine_cluster(0.5, 1.5, 0.5, 0.3), nothing, nothing)))
        rejects("polar_truncation takes a single patch without refinement",
                C.problem(bcs=(wall, per, per), domain=((0.5, 1.5), (0.0, 2π), (0.0, 1.0)),
                          metric=CylindricalMetric(), ic=C.ic_shell),
                Numerics(n_global=(48, 32, 1), execution=Execution(patch_grid=(2, 1, 1)),
                         polar_truncation=2.0))
        rejects("polar_truncation runs on the host backend only", cyl,
                Numerics(n_global=(16, 32, 1), polar_truncation=2.0,
                         execution=Execution(backend=C.device())))
        rejects("the solver components carry different floating-point types", gas,
                Numerics(n_global=(32, 1, 1), deriv=lele_d1_6(Float32)))

        # Checkpoints: a slab layout has none, and the element type and the
        # thermodynamics must match.
        dir = mktempdir()
        slabs, states = setup(gas, Numerics(n_global=n1,
                                            execution=Execution(patch_grid=(2, 1, 1))))
        @test_throws "a same-level patch layout (patch_grid) has no checkpoint" save_checkpoint(
            slabs, states, joinpath(dir, "slabs"))
        solver, Q = setup(gas, Numerics(n_global=(32, 1, 1),
                                        execution=Execution(precision=Float32)))
        save_checkpoint(solver, Q, joinpath(dir, "single"))
        solver64, Q64 = setup(gas, Numerics(n_global=(32, 1, 1)))
        @test_throws "element type mismatch" load_checkpoint!(solver64, Q64,
                                                              joinpath(dir, "single"))
        save_checkpoint(solver64, Q64, joinpath(dir, "double"))
        other, Qo = setup(C.problem(eos=IdealSpecies("gas"; R=1, gamma=1.3)),
                          Numerics(n_global=(32, 1, 1)))
        @test_throws "configuration mismatch" load_checkpoint!(other, Qo,
                                                               joinpath(dir, "double"))
    end
    println("capability matrix: $(round(time() - t0; digits=1)) s")
end
