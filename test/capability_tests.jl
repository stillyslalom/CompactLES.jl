# The combinations of docs/src/reference/capabilities.md, held against the
# setup checks. Each accepted row is built at a small size and advanced three
# steps; each rejected combination must raise the error the page quotes. A
# setup check changed without the page fails here.
#
# Standalone: julia --project=. -O1 test/capability_tests.jl
# The serial suite includes it.

using Test
using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES.Regions: Box

module Capabilities

using CompactLES
using CompactLES.Regions: Box
using CompactLES: EOS, NavierStokes1T

# An equation of state defined outside the package. It implements the methods
# the extension page lists by forwarding to an IdealMixture it holds, so its
# run is the ideal one; a method missing from that list fails here as a
# MethodError.
struct ForwardingGas{T} <: EOS
    inner::IdealMixture{T}
end
const _FG = ForwardingGas
CompactLES.nspecies(e::_FG) = CompactLES.nspecies(e.inner)
CompactLES.species_names(e::_FG) = CompactLES.species_names(e.inner)
CompactLES.recover_primitives!(s, e::_FG, Q) =
    CompactLES.recover_primitives!(s, e.inner, Q)
CompactLES.conserved_from_prim(eq::NavierStokes1T, e::_FG, pr::Prim) =
    CompactLES.conserved_from_prim(eq, e.inner, pr)
CompactLES.species_enthalpy(e::_FG, k::Int, T_ion) =
    CompactLES.species_enthalpy(e.inner, k, T_ion)
CompactLES.eos_phi(e::_FG, ρ, p, T_ion, cp) = CompactLES.eos_phi(e.inner, ρ, p, T_ion, cp)
CompactLES.eos_dphi_dY(e::_FG, k::Int, ρ, p, T_ion, cp) =
    CompactLES.eos_dphi_dY(e.inner, k, ρ, p, T_ion, cp)
CompactLES.artificial_conductivity_scale(e::_FG, ρ, c, T_ion, cp) =
    CompactLES.artificial_conductivity_scale(e.inner, ρ, c, T_ion, cp)
CompactLES.wall_internal_energy(e::_FG, Q, I, n::Int, Twall) =
    CompactLES.wall_internal_energy(e.inner, Q, I, n, Twall)
CompactLES.mole_fraction(e::_FG, k::Int, Y, I, n::Int) =
    CompactLES.mole_fraction(e.inner, k, Y, I, n)
CompactLES.state_admissibility(e::_FG, ρ, en, Yat::F, n::Int) where {F} =
    CompactLES.state_admissibility(e.inner, ρ, en, Yat, n)

const per = (PeriodicBC(), PeriodicBC())
const wall = (SlipWallBC(), SlipWallBC())
const unit = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
const cyl_domain = ((0.0, 1.0), (0.0, 2π), (0.0, 1.0))
const axis = ((AxisBC(), SlipWallBC()), per, per)

pulse(x) = 1.0 + 0.1 * exp(-40 * (x - 0.5)^2)
ic_gas(x, y, z) = Prim(p=pulse(x), rho=1.0)
ic_two(x, y, z) = (θ = 0.5 + 0.4 * sin(2π * x); Prim(Y=(θ, 1 - θ), p=1.0, rho=1.0))
ic_air(x, y, z) = Prim(p=1e5 * pulse(x), T_ion=300.0)
ic_radial(r, a, b) = Prim(p=1.0 + 0.1 * exp(-20r^2), rho=1.0)
ic_shell(r, a, b) = Prim(p=1.0 + 0.1 * exp(-20(r - 0.75)^2), rho=1.0)

const EOSES = [
    ("IdealMixture", IdealSpecies("gas"; R=1, gamma=1.4), ic_gas),
    ("IdealMixture, two species",
     IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                   IdealSpecies{Float64}("b", 0.5, 1.2)]), ic_two),
    ("StiffenedGas", StiffenedGas(gamma=4.4, p_inf=1.0, cv=1.0), ic_gas),
    ("Nasa9Mixture", Nasa9Mixture(["N2"]), ic_air),
    ("user EOS", ForwardingGas(IdealMixture([IdealSpecies{Float64}("gas", 1.0, 1.4)])),
     ic_gas),
]

device() = DeviceBackend(CompactLES.KernelAbstractions.CPU())

# The backend and precision pairs the geometry and layout rows run at; the
# EOS rows run all four.
const PAIRS = ((CompactLES.CPUBackend(), Float64), (device(), Float32))

problem(; domain=unit, bcs=(per, per, per), ic=ic_gas,
        eos=IdealSpecies("gas"; R=1, gamma=1.4), metric=CartesianMetric(),
        transport=ConstantTransport()) =
    Problem(; domain, bcs, ic, eos, metric, transport)

# Build and advance three steps. A device backend runs its pointwise bodies
# and staged exchanges through the KernelAbstractions kernels.
function advances(prob, num)
    on_device = num.backend isa DeviceBackend
    CompactLES.FORCE_KA[] = on_device
    CompactLES.FORCE_DEVICE_EXCHANGE[] = on_device
    try
        solver, Q = setup(prob, num)
        run!(solver, Q; tfinal=1.0, nmax=3)
        return solver.step == 3
    finally
        CompactLES.FORCE_KA[] = false
        CompactLES.FORCE_DEVICE_EXCHANGE[] = false
    end
end

end # module Capabilities

@testset "capability matrix" begin
    C = Capabilities
    per, wall, axis = C.per, C.wall, C.axis
    stretched = (sine_cluster(0.0, 1.0, 0.5, 0.5), nothing, nothing)
    t0 = time()

    @testset "accepted: EOS × backend × precision" begin
        for (name, eos, ic) in C.EOSES, backend in (CompactLES.CPUBackend(), C.device()),
            precision in (Float64, Float32)
            ok = C.advances(C.problem(; eos, ic),
                            Numerics(; n_global=(32, 1, 1), backend, precision))
            ok || @warn "capability matrix: $name on $backend at $precision did not advance"
            @test ok
        end
    end

    @testset "accepted: geometry × backend × precision" begin
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

    @testset "accepted: layouts × backend × precision" begin
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

    @testset "accepted: checkpoints" begin
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
                Numerics(n_global=(16, 16, 1), precision=Float32))
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
                cyl, Numerics(n_global=(16, 48, 1), patch_grid=(1, 2, 1)))
        rejects("patch decomposition across a SymmetryPlaneBC is not supported",
                C.problem(bcs=(per, per, (SymmetryPlaneBC(), SlipWallBC()))),
                Numerics(n_global=(48, 1, 16), patch_grid=(2, 1, 1)))
        rejects("patch interfaces carry closure variants for a tridiagonal filter only",
                gas, Numerics(n_global=n1, patch_grid=(2, 1, 1), filt=pyranda_filter()))
        rejects("patch interfaces support the :delta4 detector only",
                gas, Numerics(n_global=n1, patch_grid=(2, 1, 1),
                              art=ArtificialProperties(detector=:d8)))
        rejects("an explicit process grid cannot combine with patch_grid",
                gas, Numerics(n_global=n1, patch_grid=(2, 1, 1), dims=(1, 1, 1)))
        rejects("the patched dimension cannot be stretched",
                C.problem(bcs=(wall, per, per)),
                Numerics(n_global=n1, patch_grid=(2, 1, 1), stretch=stretched))
        rejects("AMR: cannot be combined with a patch_grid",
                gas, Numerics(n_global=n1, patch_grid=(2, 1, 1), amr=box))
        rejects("AMR: requires CartesianMetric", cyl, Numerics(n_global=(48, 16, 1), amr=box))
        rejects("AMR: requires a uniform grid",
                C.problem(bcs=(wall, per, per)), Numerics(n_global=n1, stretch=stretched,
                                                          amr=box))
        rejects("AMR: cannot refine a run with a SymmetryPlaneBC",
                C.problem(bcs=((SymmetryPlaneBC(), SlipWallBC()), per, per)),
                Numerics(n_global=n1, amr=box))
        rejects("AMR: regridding moves one refined level", gas,
                Numerics(n_global=n1, amr=AMR(initial=[Box((0.25, 0, 0), (0.75, 1, 1)),
                                                       Box((0.4, 0, 0), (0.6, 1, 1))],
                                              regrid_interval=1)))
        rejects("requires tile > 0 and regridding", gas,
                Numerics(n_global=n1, amr=AMR(initial=feature, max_levels=3)))
        rejects("regridding more than one refined level runs on the host backend only",
                gas, Numerics(n_global=n1, backend=C.device(),
                              amr=AMR(initial=feature, regrid_interval=1, tile=4,
                                      max_levels=3)))
        rejects("level_restriction = :filter is host-only", gas,
                Numerics(n_global=n1, backend=C.device(),
                         amr=AMR(initial=Box((0.3, 0, 0), (0.7, 1, 1)),
                                 level_restriction=:filter)))
        rejects("rebalance repartitions a tiled level at the regrid cadence", gas,
                Numerics(n_global=n1, amr=AMR(initial=feature, regrid_interval=1,
                                              rebalance=1.5)))
        rejects("interface_flux = :ghost differences through a patch or level interface",
                gas, Numerics(n_global=n1, interface_flux=:ghost))
        rejects("interface_flux = :ghost requires an unstretched CartesianMetric",
                C.problem(bcs=(wall, per, per), metric=CylindricalMetric(),
                          domain=((0.5, 1.5), (0.0, 1.0), (0.0, 1.0)), ic=C.ic_shell),
                Numerics(n_global=n1, patch_grid=(2, 1, 1), interface_flux=:ghost))
        rejects("interface_flux = :ghost with molecular transport at a refined level " *
                "supports IdealMixture, Nasa9Mixture and StiffenedGas",
                C.problem(eos=C.EOSES[5][2], transport=ConstantTransport(mu0=1e-3)),
                Numerics(n_global=n1, interface_flux=:ghost, amr=box))
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
                Numerics(n_global=(48, 32, 1), patch_grid=(2, 1, 1), polar_truncation=2.0))
        rejects("polar_truncation runs on the host backend only", cyl,
                Numerics(n_global=(16, 32, 1), polar_truncation=2.0, backend=C.device()))
        rejects("the solver components carry different floating-point types", gas,
                Numerics(n_global=(32, 1, 1), deriv=lele_d1_6(Float32)))

        # Checkpoints: a slab layout has none, and the element type and the
        # thermodynamics must match.
        dir = mktempdir()
        slabs, states = setup(gas, Numerics(n_global=n1, patch_grid=(2, 1, 1)))
        @test_throws "a same-level patch layout (patch_grid) has no checkpoint" save_checkpoint(
            slabs, states, joinpath(dir, "slabs"))
        solver, Q = setup(gas, Numerics(n_global=(32, 1, 1), precision=Float32))
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
