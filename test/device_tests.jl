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

@testset "device-resident patch: construction" begin
    cpu_ka = CL.KernelAbstractions.CPU()
    bk = DeviceBackend(cpu_ka)
    per = (PeriodicBC(), PeriodicBC())
    s = Solver(n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
               bcs=(per, per, per), backend=bk)
    @test s.deriv_plans[1] isa DevicePlan
    @test s.filter_plans[3] isa DevicePlan
    @test s.smooth_plans === s.filter_plans ||
          s.smooth_plans[1] isa Union{Nothing,DevicePlan}
    Q = allocate_state(s)
    @test Q isa ConservedState
    @test size(Q) == (size(s.rho)..., s.equations.n_cons)

    # Fold plans convert too, and the pair scratch is backend-allocated.
    sax = Solver(n_global=(16, 12, 12), L_domain=(1.0, 2π, 1.0),
                 bcs=((AxisBC(), SlipWallBC()), per, per),
                 metric=CylindricalMetric(), backend=bk)
    @test CL.fold_dplan(sax.folds[1], 1) isa DevicePlan
    @test CL.fold_fplan(sax.folds[1], -1) isa DevicePlan

    # Unsupported combinations fail at setup, not mid-run.
    @test_throws ErrorException Solver(n_global=(32, 12, 12),
        L_domain=(1.0, 1.0, 1.0), bcs=(per, per, per), backend=bk,
        patch_grid=(2, 1, 1))
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
    # and their EOS queries, plus the relaxed filter blend.
    @test compare(; nmax=8, tfinal=0.05) do backend
        s = Solver(n_global=(48, 16, 1), L_domain=(2.0, 1.0, 1.0),
                   bcs=((NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0),
                         NSCBCOutflowBC(pinf=1.0)),
                        (NoSlipWallBC(Twall=1.0), SlipWallBC()), per),
                   backend=backend, cfl=0.4, filter_cfl=0.6)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.3, 0.0, 0.0)))
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
end

@testset "device-resident patch: freestream and max_rate" begin
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())

    # Spherical origin + poles freestream: gcl_cotr! through the device plans
    # and both pair-fold parities. The discrete GCL cancels stored operator
    # output node-by-node, so this is exact on either path.
    function sph(backend)
        s = Solver(n_global=(12, 12, 12), L_domain=(1.0, Float64(π), 2π),
                   bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per),
                   metric=SphericalMetric(), backend=backend,
                   art=ArtParams(enabled=false), transport=Transport(mu0=0.0))
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, φ) -> Prim(rho=1.0, p=1.0, u=(0.0, 0.0, 0.0)))
        return s, Q
    end
    s1, Q1 = sph(CPUBackend())
    dQ1 = zero(Q1)
    compute_rhs!(s1, Q1, dQ1)
    s2, Q2 = sph(DeviceBackend(cpu_ka))
    dQ2 = zero(Q2)
    CL.FORCE_KA[] = true
    try
        compute_rhs!(s2, Q2, dQ2)
    finally
        CL.FORCE_KA[] = false
    end
    @test parent(dQ1) == parent(dQ2)

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
    # backend seam, in all three coupling modes. Bitwise against the
    # CPUBackend runs under FORCE_KA, as every device gate is. The
    # device-only write branches (`_fine_shell_point!`, the covered-region
    # and carry-over broadcasts, the staged gather pack) run on the real GPU
    # through bench/device_solver.jl.
    cpu_ka = CL.KernelAbstractions.CPU()
    per = (PeriodicBC(), PeriodicBC())
    function wave(backend; kw...)
        N = 96
        solver = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0),
                        bcs=(per, per, per), art=ArtParams(enabled=false),
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
               (subcycle=true, regrid_interval=20, tag_buffer=8))
        s1, q1 = wave(CPUBackend(); kw...)
        CL.FORCE_KA[] = true
        s2, q2 = try
            wave(DeviceBackend(cpu_ka); kw...)
        finally
            CL.FORCE_KA[] = false
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
                   bcs=(per, per, per), transport=Transport{T}(),
                   art=ArtParams{T}(), deriv=lele_d1_6(T),
                   filt=compact_filter(T(0.45), T), cfl=T(0.4),
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

# Every per-point body launched through `pointwise!`, spelled out so that a
# new one has to be added here deliberately. The scan below fails when this
# list and the call sites disagree, in either direction.
const POINTWISE_BODIES = (
    :_area_flux_point!, :_blend_interior_point!, :_body_force_point!,
    :_copy_interior_point!, :_delta4_point!, :_delta4_signed_point!,
    :_dilatation_point!, :_dilatation_switch_point!, :_extrapolation_point!,
    :_fine_shell_point!, :_fluxes_point!, :_fold_fill_point!,
    :_gate_beta_point!, :_gated_beta_point!, :_gcl_cotr_point!,
    :_grad_corr_cyl_point!, :_grad_corr_sph_point!, :_internal_energy_point!,
    :_kappa_point!, :_metric_src_cyl_point!, :_metric_src_sph_point!,
    :_mu_beta_point!, :_no_slip_wall_point!, :_nscbc_inflow_point!,
    :_nscbc_outflow_point!, :_pair_backward_local_point!,
    :_pair_backward_remote_point!, :_pair_forward_local_point!,
    :_pair_forward_remote_point!, :_pair_select_point!,
    :_primitives_ideal_point!, :_primitives_stiffened_point!, :_rate_point!,
    :_rho_sensor_point!, :_ring_accum_point!, :_rk_point!,
    :_scale_grad_point!, :_shell_ring_point!, :_slip_wall_point!,
    :_species_diffusivity_point!, :_strain_mag_point!, :_subtract_div_point!,
    :_subtract_jac_div_point!, :_zero_component_point!)

@testset "pointwise bodies stay inside the splat budget" begin
    # `_point_kernel!` calls `body!(args..., i, j, k)`. Julia expands a
    # splatted tuple of at most 32 elements into a static call and falls back
    # to the dynamic `_apply_iterate` beyond it, which is an InvalidIRError on
    # device (reference/AMR_GPU.md, pointwise kernels — the NSCBC bodies pack
    # their scalars into small tuples for exactly this reason). The budget is
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
