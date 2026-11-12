# A Solver built on a DeviceBackend is a supported configuration —
# construction converts every compact plan to a
# DevicePlan, allocates every field through the backend, and the whole step
# (boundary enforcement, folds, NSCBC, max_rate, the filter) runs through
# launchable bodies. On the KernelAbstractions CPU backend the storage is an
# ordinary Array, so these tests pin two things on one machine: that the
# DeviceBackend construction path produces a runnable solver whose line solves
# go through the DevicePlan kernels, and that a full run through those kernels
# under FORCE_KA reproduces the CPUBackend solver BITWISE — the same equality
# gate the pointwise-kernel and device-line-solve testsets carry. The
# host-staging branches an actual device takes
# (geometry upload, initialize!, Dirichlet planes) and the GPUArrays
# reductions are measured on the GPU itself by bench/device_solver.jl.

using CompactLES: compute_rhs!, max_rate, ConservedState, CPUBackend

@testset "device-resident patch: construction" begin
    cpu_ka = CL.KernelAbstractions.CPU()
    bk = DeviceBackend(cpu_ka)
    per = (PeriodicBC(), PeriodicBC())
    s = Solver(n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
               bcs=(per, per, per), backend=bk)
    @test s.deriv_plans[1] isa DevicePlan
    @test s.filter_plans[3] isa DevicePlan
    @test all(p -> p === nothing || p.closed isa DevicePlan, s.smooth_plans)
    Q = allocate_state(s)
    @test Q isa ConservedState
    @test size(Q) == (size(s.rho)..., s.equations.n_cons)

    # Fold plans convert too, and the pair scratch is backend-allocated.
    sax = Solver(n_global=(16, 12, 12), L_domain=(1.0, 2π, 1.0),
                 bcs=((AxisBC(), SlipWallBC()), per, per),
                 metric=CylindricalMetric(), backend=bk)
    @test CL.fold_dplan(sax.folds[1], 1) isa DevicePlan
    @test CL.fold_fplan(sax.folds[1], -1) isa DevicePlan

    # A face-centred symmetry plane converts the same way on the dimension it
    # sits on, here the high end of dimension 3 alone. It is self-paired, so
    # its plans are the only device objects the fold carries.
    ssym = Solver(n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=(per, per, (SlipWallBC(), SymmetryPlaneBC())),
                  backend=bk)
    @test CL.fold_dplan(ssym.folds[3], 1) isa DevicePlan
    @test CL.fold_fplan(ssym.folds[3], -1) isa DevicePlan

    # A patch layout builds on the backend too: every slab's plans are
    # device plans.
    sp = Solver(n_global=(32, 12, 12), L_domain=(1.0, 1.0, 1.0),
                bcs=(per, per, per), backend=bk, patch_grid=(2, 1, 1))
    @test all(p.deriv_plans[1] isa DevicePlan for p in sp.patches)
    @test all(p.filter_plans[1] isa DevicePlan for p in sp.patches)
end

@testset "device-resident patch layout: two slabs reproduce the CPU solver" begin
    # The same-level interface records (ghost strips, shared-plane means)
    # stage through the backend on device storage; under FORCE_KA on the KA
    # CPU backend the staged copies are the host copies element for
    # element, so the two-patch run is bitwise against the CPUBackend one.
    # The viscous case keeps the interface rows live through the gradients.
    # The layout runs C6 only: a C10 layout stages the same records, and the
    # pentadiagonal device plans are compared bitwise against the host plans
    # by the "device line solves" testset, and with the C10 interface rows
    # across a rank boundary by the MPI suite's phase of that name.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    function slabs(backend; interface_flux=:closure)
        s = Solver(n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=(per, per, per),
                   art=ArtificialProperties(enabled=false),
                   transport=ConstantTransport(mu0=2e-2),
                   patch_grid=(2, 1, 1), backend=backend,
                   interface_flux=interface_flux)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) ->
            Prim(u=(0.5 + 0.1 * sin(2x), 0, 0), p=1.0 + 0.05 * cos(x),
                 rho=1.0 + 0.2 * sin(x)))
        run!(s, states; tfinal=0.3)
        return s, states
    end
    for kw in ((;), (interface_flux=:ghost,))
        s1, q1 = slabs(CPUBackend(); kw...)
        CL.FORCE_KA[] = true
        CL.FORCE_DEVICE_EXCHANGE[] = true
        s2, q2 = try
            slabs(DeviceBackend(cpu_ka); kw...)
        finally
            CL.FORCE_KA[] = false
            CL.FORCE_DEVICE_EXCHANGE[] = false
        end
        @test s1.step == s2.step
        @test all(parent(q1[i]) == parent(q2[i]) for i in 1:2)
    end
end

@testset "device-resident patch: full runs reproduce the CPU solver" begin
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())

    # Run the same configuration on both backends; the DeviceBackend run goes
    # through the DevicePlan line solves by construction and through the KA
    # kernel bodies under FORCE_KA.
    function compare(build; nmax, tfinal)
        s1, Q1 = build(CPUBackend())
        run!(s1, Q1; tfinal=tfinal, nmax=nmax)
        s2, Q2 = build(DeviceBackend(cpu_ka))
        CL.FORCE_KA[] = true
        try
            run!(s2, Q2; tfinal=tfinal, nmax=nmax)
        finally
            CL.FORCE_KA[] = false
        end
        return s1.step == s2.step && parent(Q1) == parent(Q2)
    end

    # Multispecies closed/periodic tube: primitives, fluxes, sensors, species
    # diffusion, slip walls, the RK update, and the state filter.
    @test compare(; nmax=8, tfinal=0.02) do backend
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        s = Solver(n_global=(32, 24, 12), L_domain=(1.0, 0.6, 0.3), eos=eos,
                   bcs=((SlipWallBC(), SlipWallBC()), per, per),
                   backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.05 * cos(2π * z / 0.3)))
        end)
        return s, Q
    end

    # NSCBC inflow/outflow with an isothermal no-slip wall: the plane bodies
    # and their EOS queries, plus the relaxed filter blend. The stream varies
    # across the inflow plane so that the transverse terms of both
    # conditions are live in the launched bodies.
    @test compare(; nmax=8, tfinal=0.05) do backend
        s = Solver(n_global=(48, 16, 1), L_domain=(2.0, 1.0, 1.0),
                   bcs=((NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0),
                         NSCBCOutflowBC(pinf=1.0)),
                        (NoSlipWallBC(Twall=1.0), SlipWallBC()), per),
                   backend=backend, cfl=0.4, filter_cfl=0.6)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1 + 0.02 * cos(π * y),
                                            u=(0.3 + 0.05 * sin(π * y),
                                               0.02 * sin(π * y), 0.0)))
        return s, Q
    end

    # Resolved-θ cylindrical axis: the fold fill, pair butterfly, and select
    # bodies on the antipodal path.
    @test compare(; nmax=6, tfinal=0.02) do backend
        s = Solver(n_global=(20, 16, 12), L_domain=(1.0, 2π, 0.5),
                   bcs=((AxisBC(), SlipWallBC()), per, per),
                   metric=CylindricalMetric(), backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, z) ->
            Prim(u=(0.05r * cos(θ), 0.2r, 0.05 * sin(2π * z / 0.5)),
                 p=1.0 + 0.02r^2, rho=1.0))
        return s, Q
    end

    # Dirichlet-forced face (host-evaluated targets) with extrapolation
    # outflow.
    @test compare(; nmax=6, tfinal=0.02) do backend
        s = Solver(n_global=(32, 12, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=((DirichletBC((x, y, z, t) ->
                             Prim(rho=1.0, p=1.0 + 0.01 * sin(20t),
                                  u=(0.2, 0.0, 0.0))),
                         ExtrapolationBC()), per, per),
                   backend=backend, cfl=0.4)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.2, 0.0, 0.0)))
        return s, Q
    end

    # Between two face-centred symmetry planes: the self-paired fold's mirror
    # fill and both parity plans, with two species, the artificial properties
    # and a full-strength filter pass on every step.
    @test compare(; nmax=6, tfinal=0.02) do backend
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
        s = Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0), eos=eos,
                   bcs=(sym, per, per), backend=backend,
                   filter_interval=1, filter_cfl=0.0, cfl=0.4)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            a = 0.5 + 0.2cos(π * x)
            Prim(Y=(a, 1 - a), rho=1.0 + 0.05cos(π * x),
                 p=1.0 + 0.02cos(2π * x), u=(0.1sin(π * x), 0.0, 0.0))
        end)
        return s, Q
    end

    # The plane on dimension 2, where the fold runs on the transposed path,
    # with a tangential velocity that its even parity carries across.
    @test compare(; nmax=6, tfinal=0.02) do backend
        sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
        s = Solver(n_global=(16, 24, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(per, sym, per), backend=backend,
                   filter_interval=1, filter_cfl=0.0, cfl=0.4)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) ->
            Prim(rho=1.0 + 0.05cos(π * y), p=1.0 + 0.02cos(π * y),
                 u=(0.08cos(π * y) * sin(2π * x), 0.1sin(π * y), 0.0)))
        return s, Q
    end
end

@testset "device-resident patch: freestream and max_rate" begin
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())

    # Freestream through the device plans. On the spherical origin and poles
    # the radial and polar pressure gradients cross both paired folds; on a
    # stretched annulus the radial momentum takes its pressure gradient along
    # the stretched line; with θ collapsed, on the equator, the θ-momentum's
    # source carries no pressure. Every other flux of the state at rest is
    # zero, so the right-hand side is the compact derivative of a constant on
    # either path and the two agree bitwise.
    walls = (SlipWallBC(), SlipWallBC())
    configs = [
        (; n_global=(12, 12, 12), L_domain=(1.0, Float64(π), 2π),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per),
           metric=SphericalMetric()),
        (; n_global=(12, 12, 1), L_domain=(1.0, 2π, 1.0), origin=(0.2, 0.0, 0.0),
           bcs=(walls, per, per), metric=CylindricalMetric(),
           stretch=(sine_cluster(0.2, 1.2, 0.5, 0.4), nothing, nothing)),
        (; n_global=(12, 1, 1), L_domain=(1.0, 1.0, 1.0), origin=(0.2, π/2, 0.0),
           bcs=(walls, per, per), metric=SphericalMetric()),
    ]
    function rest(backend, config)
        s = Solver(; config..., backend=backend,
                   art=ArtificialProperties(enabled=false),
                   transport=ConstantTransport(mu0=0.0))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.0, 0.0, 0.0)))
        return s, Q
    end
    function compare_rest(config)
        s1, Q1 = rest(CPUBackend(), config)
        dQ1 = zero(Q1)
        compute_rhs!(s1, Q1, dQ1)
        s2, Q2 = rest(DeviceBackend(cpu_ka), config)
        dQ2 = zero(Q2)
        CL.FORCE_KA[] = true
        try
            compute_rhs!(s2, Q2, dQ2)
        finally
            CL.FORCE_KA[] = false
        end
        @test parent(dQ1) == parent(dQ2)
        @test maximum(abs, parent(dQ1)) < 1e-12
        return s1, Q1, s2, Q2
    end
    s1, Q1, s2, Q2 = compare_rest(configs[1])
    foreach(compare_rest, configs[2:end])

    # The launch-path rate measurement agrees with the fused serial loop
    # bitwise: same candidates, exact max/min.
    r1 = max_rate(s1, Q1)
    CL.FORCE_KA[] = true
    r2 = try
        max_rate(s2, Q2)
    finally
        CL.FORCE_KA[] = false
    end
    @test r1 == r2
end

@testset "device-resident refinement" begin
    # A refined solver on the DeviceBackend construction path: DevicePlans on
    # both levels, the level transfer's gathers and writes routed through the
    # backend interface, in all three coupling modes. Bitwise against the
    # CPUBackend runs under FORCE_KA and FORCE_DEVICE_EXCHANGE, as every
    # device gate is: the second toggle takes the device-storage branches
    # (the device chain on the LevelScratch, the shell and covered-region
    # writes, the staged gather pack, the tag sweep) on host arrays, where
    # they are the host arithmetic element for element; the same branches
    # run on the real GPU through bench/device_solver.jl.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    function wave(backend; kw...)
        N = 96
        solver = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0),
                        bcs=(per, per, per), art=ArtificialProperties(enabled=false),
                        filter_interval=0,
                        refine=BlockRegion((N ÷ 2 - N ÷ 12, 0, 0),
                                           (N ÷ 6, 1, 1)),
                        backend=backend; kw...)
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            Prim(u=(0.5, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x)))
        run!(solver, states; tfinal=0.3)
        return solver, states
    end
    for kw in ((;), (subcycle=true,),
               (subcycle=true, regrid_interval=20, tag_buffer=8),
               (subcycle=true, regrid_interval=20, tag_buffer=8,
                level_interpolation_order=8))
        s1, q1 = wave(CPUBackend(); kw...)
        CL.FORCE_KA[] = true
        CL.FORCE_DEVICE_EXCHANGE[] = true
        s2, q2 = try
            wave(DeviceBackend(cpu_ka); kw...)
        finally
            CL.FORCE_KA[] = false
            CL.FORCE_DEVICE_EXCHANGE[] = false
        end
        @test s1.step == s2.step
        @test all(parent(q1[i]) == parent(q2[i]) for i in 1:2)
    end
    # :filter restriction is host-only and refused at setup on device.
    @test_throws ErrorException Solver(n_global=(96, 1, 1),
        L_domain=(2π, 1.0, 1.0), bcs=(per, per, per),
        refine=BlockRegion((40, 0, 0), (16, 1, 1)),
        level_restriction=:filter, backend=DeviceBackend(cpu_ka))
end

@testset "device-resident refinement: the shell's admissible fallback" begin
    # A blast whose shock crosses the coarse-fine faces of a box, at the
    # global step and subcycled, inviscid and with molecular transport (whose
    # coarse-fine ghost fluxes the extrapolated remainder carries under the
    # artificial properties): shell nodes the Lagrange chain or the
    # Hermite blend leave with ρe < 0 take the multilinear interpolant, two in
    # the last step of the global run and 346 and 100 by the end of the
    # subcycled ones. Bitwise against the CPUBackend under FORCE_KA and
    # FORCE_DEVICE_EXCHANGE.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    wall = (SlipWallBC(), SlipWallBC())
    function blast(backend; nmax, kw...)
        s = Solver(; n_global=(48, 48, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(wall, wall, per), eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   art=ArtificialProperties(enabled=true), cfl=0.4,
                   control=StepControl(validity=:permissive),
                   refine=BlockRegion((20, 18, 0), (10, 10, 1)), backend=backend,
                   merge((transport=ConstantTransport(mu0=0.0),), kw)...)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                            p=1e-3 + exp(-((x - 0.5)^2 +
                                                           (y - 0.45)^2) / 0.004)))
        run!(s, Q; tfinal=0.25, nmax)
        return s, Q
    end
    for (nmax, kw) in ((40, (;)), (20, (subcycle=true,)),
                       (20, (subcycle=true, transport=ConstantTransport(mu0=1e-4))))
        s1, q1 = blast(CPUBackend(); nmax, kw...)
        CL.FORCE_KA[] = true
        CL.FORCE_DEVICE_EXCHANGE[] = true
        s2, q2 = try
            blast(DeviceBackend(cpu_ka); nmax, kw...)
        finally
            CL.FORCE_KA[] = false
            CL.FORCE_DEVICE_EXCHANGE[] = false
        end
        @test s1.step == s2.step == nmax
        @test all(parent(q1[i]) == parent(q2[i]) for i in eachindex(q1))
    end
end

@testset "device-resident refinement at a fold" begin
    # A level reaching a symmetry plane or the r-z axis on the device backend:
    # the folded box mirror uploaded to the device chain and the Hermite
    # boxes, the tile's fold plans as device plans, and on a tiled level the
    # fold's batched plans over the stack, whose mirror fill runs per tile.
    # The r-z case tiles a strip of four along the plane at z = 0 from the
    # axis outward: the corner tile folds on both dimensions and stacks alone,
    # and the other three share a stack along the active z, the plane's own
    # dimension, so its mirror fill reads the stacked launch's third index
    # modulo the tile's extent. The outermost of the three meets the parent at
    # its outer r face where the other two meet a tile, so the stack's filter
    # along r closes each tile's lines with that tile's rows. It is viscous under the default
    # ghost fluxes with the artificial properties off, so the coarse-fine
    # ghost fluxes read the gradient ring, whose folded box runs through the
    # scratch's device plans.
    # Bitwise against the CPUBackend under FORCE_KA and
    # FORCE_DEVICE_EXCHANGE, as every device gate is.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    function plane(backend)
        N = 48
        s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=((SymmetryPlaneBC(), SymmetryPlaneBC()), per, per), cfl=0.9,
                   filter_interval=1, subcycle=true,
                   refine=BlockRegion((0, 0, 0), (N ÷ 6 + 1, 1, 1)), backend=backend)
        q = allocate_state(s)
        initialize!(s, q, (x, y, z) -> begin
            rho = 1 + 0.05 * cos(pi * x)
            Prim(rho=rho, u=(0.05 * sin(pi * x), 0.0, 0.0), p=rho^1.4)
        end)
        run!(s, q; tfinal=0.05)
        return s, q
    end
    function corner(backend)
        N = 24
        s = Solver(n_global=(N, 1, N), L_domain=(1.0, 1.0, 1.0),
                   bcs=((AxisBC(), SlipWallBC()), per, (SymmetryPlaneBC(), SlipWallBC())),
                   metric=CylindricalMetric(), filter_interval=1, cfl=0.5, subcycle=true,
                   transport=ConstantTransport(mu0=1e-3),
                   art=ArtificialProperties(enabled=false),
                   refine=BlockRegion((0, 0, 0), (17, 1, 5)), tile=4, backend=backend)
        q = allocate_state(s)
        initialize!(s, q, (r, θ, z) -> begin
            rho = 1 + 0.05 * exp(-((sqrt(r^2 + z^2) - 0.4) / 0.15)^2)
            Prim(rho=rho, u=(0.02 * r, 0.0, 0.01 * z), p=rho^1.4)
        end)
        run!(s, q; tfinal=0.05, nmax=4)
        return s, q
    end
    for build in (plane, corner)
        s1, q1 = build(CPUBackend())
        CL.FORCE_KA[] = true
        CL.FORCE_DEVICE_EXCHANGE[] = true
        s2, q2 = try
            build(DeviceBackend(cpu_ka))
        finally
            CL.FORCE_KA[] = false
            CL.FORCE_DEVICE_EXCHANGE[] = false
        end
        @test s1.step == s2.step
        @test all(parent(q1[i]) == parent(q2[i]) for i in eachindex(q1))
        lt = getfield(s2, :levels)[2].transfers[1]
        @test lt.folded[1][1]
    end
    # The corner tile stacks alone and folds on r and z; the other three share
    # a stack along z that folds on z only, two kinds of r faces in it.
    s, _ = corner(DeviceBackend(cpu_ka))
    stacks = getfield(s, :levels)[2].stacks
    @test [length(st.members) for st in stacks] == [1, 3]
    trio = stacks[2]
    @test trio.patch.rho.stride > 1
    @test stacks[1].patch.folds[1].lo && stacks[1].patch.folds[3].lo
    @test !CL._fold_active(trio.patch.folds[1]) && trio.patch.folds[3].lo
    @test length(unique(CL._shared_faces(getfield(s, :patches)[i].faces)
                        for i in trio.members)) == 2
    @test CL.fold_fplan(trio.patch.folds[1], 1).rows isa CL.DeviceRowVariants
end

@testset "device-resident tiled level: regridding Sod reproduces the CPU solver" begin
    # A tiled, subcycled, regridding level on the device backend: the
    # tile records stage through the backend, the transfer chain runs on
    # the device scratch (the Hermite blend, the interpolation stages and
    # the ring pack as kernels), a fresh tile fills from the device chain,
    # and the tag sweep evaluates on the device and downloads the tag bytes.
    # Bitwise against the CPUBackend under FORCE_KA, tiles and tag history
    # included, with the artificial properties live. By t = 0.025, after 15
    # regrid checks, the regrid under the default tags has dropped two tiles
    # of the initial set and created one of them again (at check 13), so a
    # fresh tile fills from the device chain; a longer run doubles the device
    # runtime and reaches no further path.
    cpu_ka = CL.KernelAbstractions.CPU()
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    function tiled(backend; kw...)
        s = Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(wall2, per, per), cfl=0.2, subcycle=true,
                   regrid_interval=5, refine=BlockRegion((85, 0, 0), (31, 1, 1)),
                   tile=8, backend=backend; kw...)
        states = allocate_state(s)
        initialize!(s, states, ic)
        initial = level_regions(s, 1)
        run!(s, states; tfinal=0.025, nmax=400)
        return s, states, initial
    end
    for kw in ((;), (tag_sensor_threshold=0.05,),
               (interface_divergence=lele_d1_6(closures=:brady_livescu),),
               (interface_flux=:ghost,),
               (interface_flux=:ghost, transport=ConstantTransport(mu0=1e-3)))
        # The conservative coarse-fine coupling is host-only and leaves a
        # stacked device level uncorrected (src/reflux.jl), so both runs go
        # without it here.
        CL.REFLUX[] = false
        s1, q1, initial = tiled(CPUBackend(); kw...)
        CL.FORCE_KA[] = true
        CL.FORCE_DEVICE_EXCHANGE[] = true
        s2, q2, _ = try
            tiled(DeviceBackend(cpu_ka); kw...)
        finally
            CL.FORCE_KA[] = false
            CL.FORCE_DEVICE_EXCHANGE[] = false
            CL.REFLUX[] = true
        end
        @test s1.step == s2.step
        @test level_regions(s1, 1) == level_regions(s2, 1)
        @test length(level_regions(s1, 1)) > 1
        # Under the default tags the level holds a tile a regrid created.
        isempty(kw) && @test any(>(0), values(getfield(s1, :regrid).created))
        @test getfield(s1, :regrid).created == getfield(s2, :regrid).created
        @test all(parent(q1[i]) == parent(q2[i]) for i in eachindex(q1))
        # The device level's tiles take stacked storage: one stack (every
        # tile of this case has the lattice extent), whose tiles meet a tile
        # or the parent at their faces in more than one combination, the
        # stack's filter closing each tile's lines with that tile's rows; the
        # spanning patch's arrays StackedArrays over the tiles' views, the
        # states views of one stacked array in slot order, and the spanning
        # plans batched.
        lev = getfield(s2, :levels)[2]
        @test length(lev.stacks) == 1
        st = lev.stacks[1]
        @test st.members == lev.patches
        @test length(unique(CL._shared_faces(getfield(s2, :patches)[i].faces)
                            for i in lev.patches)) > 1
        @test st.patch.filter_plans[1].rows isa CL.DeviceRowVariants
        @test st.patch.rho isa CL.StackedArray
        @test st.patch.rho.ntiles == length(st.members)
        @test st.patch.deriv_plans[1].ntiles == length(st.members)
        @test st.patch.deriv_plans[1].lines == length(st.members)
        # The batched divergence plans close with the source scheme's rows.
        if haskey(kw, :interface_divergence)
            rows = kw.interface_divergence.closures
            ref = CL.plan_direction(st.patch.decomp, lele_d1_6(), 1, st.patch.h[1];
                                    lo_closures=rows, hi_closures=rows)
            @test st.patch.div_plans[1].host.chi == ref.chi
        end
        for (slot, li) in enumerate(st.members)
            tile = getfield(s2, :patches)[li]
            @test parent(tile.rho) === parent(st.patch.rho)
            @test parent(tile.rhs_workspace.tmp_a) === parent(st.patch.rhs_workspace.tmp_a)
            @test parent(parent(q2[li])) === parent(parent(q2[st.members[1]]))
            @test parentindices(parent(q2[li]))[3] ==
                  ((slot - 1) * st.patch.rho.stride + 1):(slot * st.patch.rho.stride)
        end
        @test parent(CL._stack_state(st, q2)) isa CL.StackedArray
        # A state vector assembled from separate arrays is refused.
        loose = [i == 1 ? q2[1] : ConservedState(copy(parent(q2[i]))) for i in eachindex(q2)]
        @test_throws ErrorException CL._stack_state(st, loose)
        # The host backend stacks nothing.
        @test all(isempty(lev.stacks) for lev in getfield(s1, :levels))
    end
end

@testset "device-resident tiled level: stacks along an active dimension" begin
    # The 1-D and 2-D tiled cases stack their tiles along the collapsed third
    # dimension (stride 1, no halo). A 3-D level stacks along an active one:
    # the stacked launches shift `k` by whole padded tile extents, the
    # sensor's edge clamp reduces it modulo the stride, and the halo exchange
    # along the stacking dimension runs per tile. Four tiles along z (the
    # refined box sits inside one lattice cell in x and y) in one stack, the
    # end tiles meeting the parent at one z face, so the filter along z closes
    # the tiles' lines with three sets of rows; subcycled so the filter and
    # the Hermite shell batch too, one step,
    # bitwise against the CPU hierarchy. The cost of this testset is the
    # compilation of the 3-D refined drivers on both backends, so it is kept
    # to four tiles and a step; `bench/device_solver.jl` runs the twelve-tile
    # case.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    function box3d(backend)
        N = 36
        s = Solver(n_global=(N, N, N), L_domain=(2π, 2π, 2π), bcs=(per, per, per),
                   transport=ConstantTransport(mu0=1e-2), subcycle=true,
                   refine=BlockRegion((12, 12, 6), (7, 7, 25)), tile=6,
                   backend=backend)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) ->
            Prim(u=(0.3 + 0.1 * sin(x) * cos(y) * cos(z),
                    -0.1 * cos(x) * sin(y) * cos(z), 0.05 * sin(z)),
                 p=1.0 + 0.02 * cos(2x) * cos(z), rho=1.0 + 0.1 * sin(x + y + z)))
        run!(s, states; tfinal=0.2, nmax=1)
        return s, states
    end
    s1, q1 = box3d(CPUBackend())
    CL.FORCE_KA[] = true
    CL.FORCE_DEVICE_EXCHANGE[] = true
    s2, q2 = try
        box3d(DeviceBackend(cpu_ka))
    finally
        CL.FORCE_KA[] = false
        CL.FORCE_DEVICE_EXCHANGE[] = false
    end
    lev = getfield(s2, :levels)[2]
    @test length(lev.stacks) == 1 && length(lev.stacks[1].members) == 4
    @test lev.stacks[1].patch.rho.stride > 1
    @test length(lev.stacks[1].patch.filter_plans[3].rows.nclo) == 3
    @test s1.step == s2.step
    @test all(parent(q1[i]) == parent(q2[i]) for i in eachindex(q1))
end

@testset "device-resident tiled level: the bulk species channel" begin
    # A two-species planar slab under `species_flux = :bulk` on a tiled level
    # of four tiles in a row: the component copy, the conserved gradients
    # through the batched plans, the mole-fraction and maximum passes and the
    # bulk flux all run once over the spanning patch's arrays (one stack, the
    # filter along x closing its tiles' lines with three sets of rows),
    # bitwise against the CPU hierarchy after two steps.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 1 / 20, 1.4)])
    function slab2d(backend)
        N = 48
        h = 1.0 / N
        s = Solver(n_global=(N, N ÷ 2, 1), L_domain=(1.0, 0.5, 1.0),
                   bcs=(per, per, per), eos=eos,
                   art=ArtificialProperties(enabled=true, species_flux=:bulk),
                   transport=ConstantTransport(mu0=0.0), filter_interval=1,
                   refine=BlockRegion((18, 12, 0), (25, 7, 1)), tile=6,
                   backend=backend)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) -> begin
            V = (1 - tanh((abs(x - 0.5) - 0.25) / 4h)) / 2
            ρ = V * 20 + (1 - V)
            Prim(Y=(1 - V * 20 / ρ, V * 20 / ρ), rho=ρ, u=(1.0, 0.3, 0.0), p=1.0)
        end)
        run!(s, states; tfinal=1.0, nmax=2)
        return s, states
    end
    # Without the host-only conservative coupling, as above.
    CL.REFLUX[] = false
    s1, q1 = slab2d(CPUBackend())
    CL.FORCE_KA[] = true
    CL.FORCE_DEVICE_EXCHANGE[] = true
    s2, q2 = try
        slab2d(DeviceBackend(cpu_ka))
    finally
        CL.FORCE_KA[] = false
        CL.FORCE_DEVICE_EXCHANGE[] = false
        CL.REFLUX[] = true
    end
    lev = getfield(s2, :levels)[2]
    @test length(lev.stacks) == 1 && length(lev.stacks[1].members) == 4
    @test !isempty(lev.stacks[1].patch.rhs_workspace.grad_Q)
    @test length(lev.stacks[1].patch.filter_plans[1].rows.nclo) == 3
    @test s1.step == s2.step
    @test all(parent(q1[i]) == parent(q2[i]) for i in eachindex(q1))
    @test all(maximum(Array(p.D_art[1])) > 0 for p in getfield(s2, :patches))
end

@testset "launch policy toggle" begin
    # DEVICE_SYNC[] switches between the deferred default and the
    # synchronize-per-launch fallback; both must produce identical states.
    # On the KA CPU backend kernels run synchronously either way, so this
    # pins the toggle's plumbing; the GPU delta is bench/device_solver.jl's
    # sync= comparison.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    function short(backend)
        s = Solver(n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                   bcs=(per, per, per), backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) ->
            Prim(rho=1 + 0.05sin(2π * x), p=1.0, u=(0.1, 0.0, 0.0)))
        run!(s, Q; tfinal=0.05, nmax=4)
        return parent(Q)
    end
    @test CL.DEVICE_SYNC[] == false
    CL.FORCE_KA[] = true
    a = try
        short(DeviceBackend(cpu_ka))
    finally
        CL.FORCE_KA[] = false
    end
    CL.FORCE_KA[] = true
    CL.DEVICE_SYNC[] = true
    b = try
        short(DeviceBackend(cpu_ka))
    finally
        CL.FORCE_KA[] = false
        CL.DEVICE_SYNC[] = false
    end
    @test a == b
end

@testset "device-resident patch: Float32 step" begin
    # Float32 through the device construction path, at test scale: a
    # Float32 solver on the
    # DeviceBackend construction path advances and stays finite, bitwise
    # against the Float32 CPU solver.
    cpu_ka = CL.KernelAbstractions.CPU()
    T = Float32
    per = (PeriodicBC(), PeriodicBC())
    function build(backend)
        s = Solver(n_global=(16, 12, 12), L_domain=(T(2π), T(2π), T(2π)),
                   bcs=(per, per, per), precision=T, cfl=0.4,
                   backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) ->
            Prim(rho=1 + 0.01sin(x), T_ion=1, u=(0.1cos(y), 0.05sin(z), 0.0)))
        return s, Q
    end
    s1, Q1 = build(CPUBackend())
    run!(s1, Q1; tfinal=T(0.05), nmax=5)
    s2, Q2 = build(DeviceBackend(cpu_ka))
    @test s2.deriv_plans[1] isa DevicePlan{T}
    @test s2.eos isa IdealMixture{T} && s2.art isa ArtificialProperties{T}
    CL.FORCE_KA[] = true
    try
        run!(s2, Q2; tfinal=T(0.05), nmax=5)
    finally
        CL.FORCE_KA[] = false
    end
    @test s1.step == s2.step
    @test all(isfinite, parent(Q2))
    @test parent(Q1) == parent(Q2)
end

# Everything a device launch asks of a NASA-9 EOS at one state, written as a
# pointwise body so that the launch carries the EOS as a kernel argument: the
# recovery and its status, the admissibility verdict, and the species and
# mixture evaluations at a given temperature, in both the scalar and the
# shared-powers forms.
function _nasa9_probe_point!(out, eos, e, Y, Tq, n_species, i, j, k)
    @inbounds begin
        Yat = sp -> Y[sp, i]
        T_rec, status = CL.mixture_temperature_status(eos, e[i], Yat)
        out[1, i] = T_rec
        out[2, i] = status
        out[3, i] = state_admissibility(eos, one(T_rec), e[i], Yat, n_species)
        T = Tq[i]
        point = CL._species_point(eos, T)
        cp = zero(T); h = zero(T); ev = zero(T); hp = zero(T); ep = zero(T)
        Rm = zero(T)
        for sp in 1:n_species
            cp += Y[sp, i] * CL.species_cp(eos, sp, T)
            h += Y[sp, i] * CL.species_enthalpy(eos, sp, T)
            ev += Y[sp, i] * CL.species_energy(eos, sp, T)
            hp += Y[sp, i] * CL.species_enthalpy(eos, sp, point)
            ep += Y[sp, i] * CL.species_energy(eos, sp, point)
            Rm += Y[sp, i] * eos.Rk[sp]
        end
        out[4, i] = cp; out[5, i] = h; out[6, i] = ev; out[7, i] = hp
        out[8, i] = ep
        p = Rm * T
        out[9, i] = CL.eos_phi(eos, one(T), p, T, cp)
        out[10, i] = CL.eos_dphi_dY(eos, n_species, one(T), p, T, cp)
        out[11, i] = CL.artificial_conductivity_scale(eos, one(T), one(T), T, cp)
    end
    return nothing
end

@testset "NASA-9 device mirror: recovery and status" begin
    # A device launch replaces a Nasa9Mixture with its isbits mirror, and on
    # the KA CPU backend `_kernel_arg` does the same, so the launch below runs
    # the mirror's methods. They must reproduce the host mixture bit for bit
    # over the states the inversion finds hard: interval joins approached to
    # a few ulp, the edges of the fitted range and beyond them under each
    # extrapolation policy, energies with no root in the search bounds, and
    # compositions with no bracket.
    cpu_ka = CL.KernelAbstractions.CPU()
    for T in (Float64, Float32), policy in (:polynomial, :linear, :missing)
        eos = Nasa9Mixture(T, ["He", "CO2", "N2"]; extrapolate=policy)
        mirror = CL.Adapt.adapt(cpu_ka, eos)
        @test mirror isa CL.Nasa9MixtureCoeffs{T,3}
        @test isbitstype(typeof(mirror))
        @test size(mirror.intervals) == size(eos.intervals)
        @test all(mirror.intervals[i, k] === eos.intervals[i, k]
                  for i in axes(eos.intervals, 1), k in 1:3)
        comps = (T.((0.2, 0.5, 0.3)), T.((1, 0, 0)), T.((0, 1, 0)),
                 T.((-0.3, 0.8, 0.5)), T.((0, 0, 0)))
        temps = T[]
        for Tj in (1000, 6000), s in (-1e-3, -4eps(T), 0, 4eps(T), 1e-3)
            push!(temps, T(Tj) * (1 + T(s)))
        end
        append!(temps, T.((100, 200, 250, 300, 2500, 20000, 30000, 1e5)))
        states = [(Y, Tr, sum(Y[k] * CL.species_energy(eos, k, Tr) for k in 1:3))
                  for Y in comps for Tr in temps]
        for Y in comps, e in T.((-1e12, 1e30))
            push!(states, (Y, T(500), e))
        end
        n = length(states)
        Ym = T[states[i][1][k] for k in 1:3, i in 1:n]
        Tq = T[s[2] for s in states]
        e = T[s[3] for s in states]
        host = zeros(T, 11, n)
        for i in 1:n
            _nasa9_probe_point!(host, eos, e, Ym, Tq, 3, i, 1, 1)
        end
        dev = zeros(T, 11, n)
        CL.pointwise_ka!(_nasa9_probe_point!, cpu_ka, n, 1, 1, dev, eos, e, Ym,
                         Tq, 3)
        @test isequal(host, dev)
        # The set reaches the failure and extrapolation statuses.
        hard = CL.TEMPERATURE_NOT_CONVERGED | CL.TEMPERATURE_NO_BRACKET |
               CL.TEMPERATURE_OUT_OF_RANGE
        @test reduce(|, UInt8.(host[2, :])) & hard == hard
        verdicts = reduce(|, UInt8.(host[3, :]))
        @test verdicts & CL.STATE_EXTRAPOLATED != 0
        @test (verdicts & CL.STATE_INADMISSIBLE != 0) == (policy === :missing)
    end
end

@testset "NASA-9 device mirror: full runs reproduce the CPU solver" begin
    # A NASA-9 run on the DeviceBackend construction path under FORCE_KA,
    # bitwise against the CPUBackend run. The gas crosses the 1000 K join of
    # both fits between the hot core and the isothermal wall, so the
    # recovery, the fluxes, the sensors' mole fractions and the wall's
    # internal energy all evaluate on both sides of it through the mirror.
    # The run is Float64 only: the Float32 mirror is compared bitwise against
    # the host mixture in every evaluation a launch makes by the recovery
    # testset above, and a Float32 step on the device path by the Float32
    # step testset.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    function wall_case(backend)
        eos = Nasa9Mixture(["N2", "CO2"])
        s = Solver(n_global=(32, 12, 1), L_domain=(0.1, 0.04, 1.0),
                   eos=eos, backend=backend, cfl=0.4,
                   bcs=((NoSlipWallBC(Twall=800.0), SlipWallBC()), per, per),
                   transport=ConstantTransport(mu0=2e-5))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = (1 + tanh((x - 0.05) / 0.008)) / 2
            Prim(Y=(1 - θ, θ), p=1e5, T_ion=900 + 700 * sin(π * x / 0.1),
                 u=(0.0, 20 * sin(2π * y / 0.04), 0.0))
        end)
        return s, Q
    end
    # A refined periodic wave with the ghost interface flux: the coarse-fine
    # molecular flux recovers the temperature gradient from the species
    # internal energies at the interface.
    function level_case(backend)
        N = 72
        eos = Nasa9Mixture(["N2", "CO2"])
        s = Solver(n_global=(N, 1, 1), L_domain=(0.1, 1.0, 1.0), eos=eos,
                   bcs=(per, per, per), transport=ConstantTransport(mu0=2e-5),
                   refine=BlockRegion((N ÷ 2 - N ÷ 12, 0, 0), (N ÷ 6, 1, 1)),
                   subcycle=true, interface_flux=:ghost, backend=backend)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) -> begin
            θ = (1 + sin(2π * x / 0.1)) / 2
            Prim(Y=(1 - θ, θ), p=1e5, T_ion=700 + 600θ, u=(30.0, 0, 0))
        end)
        return s, states
    end
    function compare(build; nmax)
        s1, q1 = build(CPUBackend())
        run!(s1, q1; tfinal=1.0, nmax=nmax)
        CL.FORCE_KA[] = true
        CL.FORCE_DEVICE_EXCHANGE[] = true
        s2, q2 = try
            s, q = build(DeviceBackend(cpu_ka))
            run!(s, q; tfinal=1.0, nmax=nmax)
            s, q
        finally
            CL.FORCE_KA[] = false
            CL.FORCE_DEVICE_EXCHANGE[] = false
        end
        a = q1 isa AbstractVector ? q1 : [q1]
        b = q2 isa AbstractVector ? q2 : [q2]
        return s1.step == s2.step == nmax && all(isfinite, parent(b[1])) &&
               all(parent(a[i]) == parent(b[i]) for i in eachindex(a))
    end
    @test compare(wall_case; nmax=6)
    @test compare(level_case; nmax=4)
end

@testset "device-resident positivity limiter reproduces the CPU solver" begin
    # Limited runs on the DeviceBackend construction path, bitwise against the
    # CPUBackend under FORCE_KA and FORCE_DEVICE_EXCHANGE: the limiter's
    # bodies as kernels over arrays allocated in the fields' storage, and the
    # host steps through their staged copies (the tallies of each θ pass, the
    # minima that set ε, and on two patches the θ exchange at the interface).
    # Each case limits stage and filter faces within its steps; the state,
    # the step count, the time and the limiter's counts agree. The cases are
    # the strong shocks of positivity_tests.jl at test scale: Woodward–Colella
    # on one patch, on two and through a refined box at the global step and
    # subcycled, the last two under either interface flux, a planar blast at a
    # wall, the Sedov blast on the spherical line folded at the origin, and on
    # the r-z plane at the corner of the axis and a symmetry plane.
    cpu_ka = CL.KernelAbstractions.CPU()
    gas = IdealSpecies("gas"; gamma=1.4, R=1.0)
    per = (PeriodicBC(), PeriodicBC())
    shock(backend; N=120, nmax=40, patches=1, amr=nothing, flux=:closure) = begin
        h = 1.0 / (N - 1)
        prob = Problem(eos=gas, transport=ConstantTransport(mu0=0.0),
                       domain=((0.0, 1.0), (0.0, h), (0.0, h)),
                       bcs=((SlipWallBC(), SlipWallBC()), per, per),
                       ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                            p=1000 * (1 - tanh_blend(x, 0.1, 2h)) +
                              0.01 * (tanh_blend(x, 0.1, 2h) - tanh_blend(x, 0.9, 2h)) +
                              100 * tanh_blend(x, 0.9, 2h)))
        setup(prob, Numerics(n_global=(N, 1, 1), cfl=0.3,
                             art=ArtificialProperties(enabled=true),
                             filter=StateFilter(compact_filter(); cfl=0.35),
                             control=StepControl(validity=:permissive),
                             execution=Execution(storage=backend,
                                                 patch_grid=(patches, 1, 1)),
                             patch_interfaces=flux, amr=amr,
                             positivity_limiter=true))
    end
    blast(backend) = setup(
        Problem(eos=gas, transport=ConstantTransport(mu0=0.0),
                domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((SlipWallBC(), SlipWallBC()), per, per),
                ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                     p=1e-3 + exp(-((x - 0.3)^2 + (y - 0.5)^2) / 0.004))),
        Numerics(n_global=(32, 32, 1), cfl=0.4, art=ArtificialProperties(enabled=true),
                 control=StepControl(validity=:permissive),
                 execution=Execution(storage=backend), positivity_limiter=true))
    sedov(backend) = setup(
        Problem(eos=gas, transport=ConstantTransport(mu0=0.0), metric=SphericalMetric(),
                domain=((0.0, 1.0), (π / 2, π / 2 + 1), (0.0, 1.0)),
                bcs=((OriginBC(), SlipWallBC()), per, per),
                ic=(r, a, b) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                     p=1e-4 + exp(-(r / 0.08)^2))),
        Numerics(n_global=(64, 1, 1), cfl=0.3, art=ArtificialProperties(enabled=true),
                 control=StepControl(validity=:permissive),
                 execution=Execution(storage=backend), positivity_limiter=true))
    corner(backend) = setup(
        Problem(eos=gas, transport=ConstantTransport(mu0=0.0), metric=CylindricalMetric(),
                domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((AxisBC(), SlipWallBC()), per, (SymmetryPlaneBC(), SlipWallBC())),
                ic=(r, θ, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                     p=1e-4 + exp(-(r^2 + z^2) / 0.08^2))),
        Numerics(n_global=(32, 1, 32), cfl=0.3, art=ArtificialProperties(enabled=true),
                 control=StepControl(validity=:permissive),
                 execution=Execution(storage=backend), positivity_limiter=true))
    # Over the discontinuity at x = 0.9, clear of the wall.
    box = BlockRegion((96, 0, 0), (20, 1, 1))
    # A pressure jump beside the interface of two patches, whose shock limits
    # the faces about the shared node from one side first, so that the
    # exchange lowers θ on the other.
    patched(backend; flux=:closure) = setup(
        Problem(eos=gas, transport=ConstantTransport(mu0=0.0),
                domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((SlipWallBC(), SlipWallBC()), per, per),
                ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                     p=1000 * (1 - tanh_blend(x, 0.45, 2 / 95)) + 0.01)),
        Numerics(n_global=(96, 1, 1), cfl=0.3, art=ArtificialProperties(enabled=true),
                 filter=StateFilter(compact_filter(); cfl=0.35),
                 control=StepControl(validity=:permissive),
                 execution=Execution(storage=backend, patch_grid=(2, 1, 1)),
                 patch_interfaces=flux, positivity_limiter=true))
    # A :ghost case's build, which keeps the limited interfaces at :ghost
    # (`LIMITED_GHOST_INTERFACES`, off by default).
    kept(build) = backend -> begin
        CL.LIMITED_GHOST_INTERFACES[] = true
        try
            build(backend)
        finally
            CL.LIMITED_GHOST_INTERFACES[] = false
        end
    end
    cases = (("one patch", b -> shock(b), 40), ("planar blast", blast, 8),
             ("spherical line", sedov, 60), ("r-z corner", corner, 12),
             ("two patches", patched, 30),
             ("refined box", b -> shock(b; amr=AMR(initial=box)), 30),
             ("refined box, subcycled",
              b -> shock(b; amr=AMR(initial=box, subcycle=true)), 20),
             ("two patches, :ghost", kept(b -> patched(b; flux=:ghost)), 30),
             ("refined box, :ghost",
              kept(b -> shock(b; amr=AMR(initial=box), flux=:ghost)), 30),
             ("refined box, subcycled, :ghost",
              kept(b -> shock(b; amr=AMR(initial=box, subcycle=true), flux=:ghost)), 20))
    for (name, build, nmax) in cases
        s1, q1 = build(CPUBackend())
        run!(s1, q1; tfinal=1.0, nmax=nmax)
        CL.FORCE_KA[] = true
        CL.FORCE_DEVICE_EXCHANGE[] = true
        s2, q2 = try
            s, q = build(DeviceBackend(cpu_ka))
            run!(s, q; tfinal=1.0, nmax=nmax)
            s, q
        finally
            CL.FORCE_KA[] = false
            CL.FORCE_DEVICE_EXCHANGE[] = false
        end
        a = q1 isa AbstractVector ? q1 : [q1]
        b = q2 isa AbstractVector ? q2 : [q2]
        counts = CL.positivity_counts(s1)
        @testset "$name" begin
            @test counts.stage_limited > 0 && counts.filter_limited > 0
            @test s1.step == s2.step == nmax && s1.t == s2.t
            @test CL.positivity_counts(s2) == counts
            @test all(parent(a[i]) == parent(b[i]) for i in eachindex(a))
        end
    end
end

@testset "a limited fill under :ghost reproduces the CPU solver" begin
    # positivity_tests.jl's fill ahead of a jump into cold gas, on a box at the
    # axis over the first 25 root nodes filled again from the root as a regrid
    # fills a new box: the nodes the interpolation leaves inadmissible take
    # the multilinear state of their parent nodes in a kernel, bitwise against
    # the CPUBackend under FORCE_KA and FORCE_DEVICE_EXCHANGE.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    prob = Problem(eos=IdealSpecies("gas"; gamma=5 / 3, R=1.0),
                   transport=ConstantTransport(mu0=0.0), metric=CylindricalMetric(),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=((AxisBC(), SlipWallBC()), per, per),
                   ic=(r, θ, z) -> r < 0.15 ? Prim(rho=16.0, u=(0.0, 0.0, 0.0), p=16 / 3) :
                                              Prim(rho=1.0, u=(-1.0, 0.0, 0.0), p=1e-4))
    filled(backend) = begin
        CL.LIMITED_GHOST_INTERFACES[] = true
        s, Q = try
            setup(prob, Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=true),
                                 execution=Execution(storage=backend),
                                 patch_interfaces=:ghost, positivity_limiter=true,
                                 amr=AMR(initial=BlockRegion((0, 0, 0), (25, 1, 1)))))
        finally
            CL.LIMITED_GHOST_INTERFACES[] = false
        end
        CL._fill_tiles_from_parent!(s, Q, getfield(s, :levels)[2])
        return s, Q
    end
    s1, q1 = filled(CPUBackend())
    CL.FORCE_KA[] = true
    CL.FORCE_DEVICE_EXCHANGE[] = true
    s2, q2 = try
        filled(DeviceBackend(cpu_ka))
    finally
        CL.FORCE_KA[] = false
        CL.FORCE_DEVICE_EXCHANGE[] = false
    end
    report = state_report(s1, q1)
    @test s1.interface_flux === s2.interface_flux === :ghost
    @test report.inadmissible == 0 && report.negative_density == 0
    @test all(parent(q1[i]) == parent(q2[i]) for i in eachindex(q1))
end

@testset "mole fractions on device storage match the host" begin
    # `:X` is computed point by point on host copies of the mass fractions and
    # into a host array, so a device-storage solver gives the host's values
    # bitwise, through the snapshot, the padded array and a VTK piece, on one
    # patch and on a tiled level whose tiles hold views of stacked storage.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)])
    mix(x) = (θ = 0.5 * (1 + tanh((x - 0.5) / 0.05)); (1 - θ, θ))
    function tube(backend)
        s = Solver(n_global=(32, 24, 12), L_domain=(1.0, 0.6, 0.3), eos=eos,
                   bcs=((SlipWallBC(), SlipWallBC()), per, per), backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(Y=mix(x), rho=1.0, p=1.0))
        return s, Q
    end
    function tiled(backend)
        s = Solver(n_global=(48, 24, 1), L_domain=(1.0, 0.5, 1.0), eos=eos,
                   bcs=(per, per, per), refine=BlockRegion((18, 8, 0), (12, 8, 1)),
                   tile=6, backend=backend)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) -> Prim(Y=mix(x), rho=1.0, p=1.0))
        return s, states
    end
    dir = mktempdir()
    s1, q1 = tube(CPUBackend())
    s2, q2 = tube(DeviceBackend(cpu_ka))
    t1, r1 = tiled(CPUBackend())
    t2, r2 = tiled(DeviceBackend(cpu_ka))
    host = field_snapshot(s1, q1; fields=(:X,))
    host_tiles = field_snapshot(t1, r1; fields=(:X,))
    save_vtk(s1, q1, joinpath(dir, "host"); fields=(:X,))
    CL.FORCE_KA[] = true
    dev, dev_tiles, padded = try
        save_vtk(s2, q2, joinpath(dir, "dev"); fields=(:X,))
        (field_snapshot(s2, q2; fields=(:X,)), field_snapshot(t2, r2; fields=(:X,)),
         field_array(s2, q2, :X; species=2))
    finally
        CL.FORCE_KA[] = false
    end
    @test dev[:X] isa Array && dev[:X] == host[:X]
    @test maximum(abs, sum(host[:X]; dims=4) .- 1) < 1e-14
    @test padded isa Array && padded == field_array(s1, q1, :X; species=2)
    @test read(joinpath(dir, "dev.r0000.vtr")) == read(joinpath(dir, "host.r0000.vtr"))
    lev = getfield(t2, :levels)[2]
    @test sum(st -> length(st.members), lev.stacks) == length(lev.patches) == 4
    @test length(dev_tiles) == length(host_tiles) == 5
    @test all(d[:X] == h[:X] for (d, h) in zip(dev_tiles, host_tiles))
    rm(dir; recursive=true)
end

@testset "field collections adapt to their device forms" begin
    # DeviceFieldVector, DeviceFieldMatrix and the ConservedState rule are
    # what a real device launch builds out of the host wrappers. Nothing
    # above reaches them: the KernelAbstractions CPU backend adapts no kernel
    # argument, so neither FORCE_KA nor the DeviceBackend runs constructs one.
    # Apply the rules directly instead, to the collections a patch carries.
    cpu_ka = CL.KernelAbstractions.CPU()
    adapt = CL.Adapt.adapt
    per = (PeriodicBC(), PeriodicBC())
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)])
    s = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0), eos=eos,
               bcs=(per, per, per))
    ft = s.field_tuples

    for name in (:Y, :D_art)
        w = getfield(ft, name)
        d = adapt(cpu_ka, w)
        @test d isa CL.DeviceFieldVector
        @test length(d) == length(w) > 1
        # Identity, not equality: the tuple must hold the patch's own arrays.
        @test all(d[i] === w[i] for i in 1:length(w))
    end
    for name in (:grad_u, :grad_Y, :flux)
        w = getfield(ft, name)
        d = adapt(cpu_ka, w)
        @test d isa CL.DeviceFieldMatrix
        n1, n2 = size(w)
        @test size(d) == (n1, n2)
        @test all(d[a, b] === w[a, b] for a in 1:n1, b in 1:n2)
    end
    # The conserved state adapts to the same wrapper around the adapted array,
    # so a body still indexes Q[I, c] on either path.
    Q = allocate_state(s)
    dQ = adapt(cpu_ka, Q)
    @test dQ isa ConservedState
    @test parent(dQ) === parent(Q)
end

# Every per-point body launched through `pointwise!` is spelled out. A new one
# must be added here deliberately. The scan below fails when this
# list and the call sites disagree, in either direction.
const POINTWISE_BODIES = (
    :_ablation_apply_point!, :_ablation_bin_driver_point!, :_ablation_bin_force_point!,
    :_ablation_bin_layer_point!, :_ablation_bin_marker_point!, :_ablation_driver_point!,
    :_ablation_front_point!, :_ablation_layer_point!,
    :_area_flux_less_point!, :_area_flux_point!, :_blend_interior_point!,
    :_body_force_point!,
    :_bulk_flux_point!, :_child_derivative_mask_point!, :_child_mask_point!,
    :_child_residual_point!, :_child_source_point!,
    :_coarse_fine_flux_point!,
    :_copy_component_point!,
    :_copy_interior_point!, :_delta4_interior_point!, :_delta4_point!,
    :_delta4_signed_point!,
    :_dilatation_point!, :_dilatation_switch_point!, :_edge_fill_point!,
    :_edge_fill_z_point!, :_energy_split_point!, :_extrapolation_point!,
    :_face_pick_point!,
    :_face_put_point!, :_face_save_point!, :_fill_fallback_point!,
    :_fine_shell_point!, :_fluxes_point!, :_fold_fill_point!, :_fold_fill_z_point!,
    :_gate_beta_point!, :_gated_beta_point!,
    :_grad_corr_cyl_point!, :_grad_corr_sph_point!, :_hermite_point!,
    :_internal_energy_point!, :_interp_point!, :_inviscid_flux_point!,
    :_kappa_point!, :_limiter_anchor_point!, :_limiter_correct_point!,
    :_limiter_end_anchor_point!, :_limiter_end_face_point!,
    :_limiter_offset_point!, :_limiter_radial_correct_point!,
    :_limiter_radial_rate_point!, :_limiter_radial_theta_point!,
    :_limiter_scan_point!, :_limiter_subtract_point!, :_limiter_theta_point!,
    :_max_into_point!, :_metric_src_cyl_point!,
    :_metric_src_sph_point!, :_molecular_flux_point!, :_mole_fraction_point!,
    :_momentum_split_point!, :_flux_store_point!, :_extrapolate_ghost_point!,
    :_ghost_flux_copy_point!,
    :_mu_beta_point!, :_no_slip_flux_point!, :_no_slip_wall_point!, :_nscbc_inflow_point!,
    :_nscbc_outflow_point!, :_pair_backward_local_point!,
    :_pair_backward_remote_point!, :_pair_forward_local_point!,
    :_pair_forward_remote_point!, :_pair_select_point!, :_partial_density_flux_point!,
    :_primitives_ideal_point!, :_primitives_nasa9_point!, :_primitives_stiffened_point!,
    :_rate_point!,
    :_reciprocal_interior_point!, :_reciprocal_point!,
    :_rho_sensor_point!, :_ring_accum_point!, :_ring_pack_field_point!,
    :_ring_pack_point!, :_ring_unpack_point!, :_rk_point!,
    :_scale_grad_point!, :_scale_interior_point!, :_sharpen_flux_point!,
    :_shell_fallback_point!, :_shell_linear_point!, :_shell_ring_point!,
    :_shell_select_point!,
    :_slip_flux_point!, :_slip_wall_point!,
    :_species_bound_point!, :_species_diffusivity_point!, :_species_split_point!,
    :_strain_mag_point!,
    :_subtract_div_point!, :_subtract_interior_point!,
    :_subtract_jac_div_point!, :_tag_delta4_point!, :_tag_gradient_point!,
    :_tag_rho_point!, :_tag_sensor_point!, :_tag_vorticity_point!,
    :_transport_status_point!,
    :_volume_fraction_point!, :_volume_weight_point!, :_weighted_blend_point!,
    :_zero_component_point!)

@testset "pointwise bodies stay inside the splat budget" begin
    # `_point_kernel!` calls `body!(args..., i, j, k)`. Julia expands a
    # splatted tuple of at most 32 elements into a static call and falls back
    # to the dynamic `_apply_iterate` beyond it, which is an InvalidIRError on
    # device (reference/AMR_GPU.md, pointwise kernels — the NSCBC bodies pack
    # their scalars into small tuples to stay below this limit). The budget is
    # therefore 32 launcher arguments, or 35 declared arguments once i, j, k
    # are counted. `_rate_point!` and `_fluxes_point!` are the two that have
    # come nearest it; both pack scalars into isbits tuples to stay under.
    src = joinpath(@__DIR__, "..", "src")
    found = Set{Symbol}()
    for f in readdir(src; join=true)
        endswith(f, ".jl") || continue
        for m in eachmatch(r"pointwise!\(\s*(_[A-Za-z0-9_]*point!)",
                           read(f, String))
            push!(found, Symbol(m.captures[1]))
        end
    end
    @test sort(collect(found)) == sort(collect(POINTWISE_BODIES))

    arity(f) = maximum(m -> m.nargs, methods(f)) - 1
    over = [(n, arity(getfield(CL, n)) - 3) for n in POINTWISE_BODIES
            if arity(getfield(CL, n)) - 3 > 32]
    @test isempty(over)
    @test arity(CL._rate_point!) - 3 <= 32
    @test arity(CL._fluxes_point!) - 3 <= 32
end
