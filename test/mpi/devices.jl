# The MPI suite's phases of the device line solves and the staged exchange. Included
# by test/mpi_tests.jl, which loads the package, defines `check`, `section` and the
# other helpers used here, and runs the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# Device line solves (reference/AMR_GPU.md). A DevicePlan runs the fill,
# sweep, spike correction and scatter as KernelAbstractions kernels and keeps
# the reduced interface stage on the host, mirroring the host arithmetic per
# line operation for operation — so on the KA CPU backend the decomposed
# comparison is BITWISE, not to round-off. Splitting each dimension in turn
# forces the cross-rank Allgather of the reduced stage through the device
# path's pack/unpack staging, for both the tridiagonal C6 (periodic wrap) and
# the pentadiagonal C10 (closed edge ranks carrying closure rows).
# ---------------------------------------------------------------------------
function test_device_lines()
    section("device line solves: DevicePlan bitwise vs host under decomposition")
    cpu = KernelAbstractions.CPU()
    # The third case closes the C10 line with its interface rows, which read
    # the ghost layers: the padded field carries the function into its
    # edge halos so the rows have data to read on the edge ranks.
    for (label, scheme, periodic, interface) in
            (("C6 periodic", CL.lele_d1_6(Float64), true, false),
             ("C10 closed", CL.lele_d1_10(Float64), false, false),
             ("C10 interface rows", CL.lele_d1_10(Float64), false, true))
        for ax in 1:3
            n_global = ntuple(d -> d == ax ? SPLITN : 12, 3)
            decomp = Decomp(n_global, (periodic, periodic, periodic);
                            dims=splitdims(ax))
            rows = interface ? CL.interface_closures(scheme) : nothing
            plan = CL.plan_direction(decomp, scheme, ax, 0.05;
                                     lo_closures=rows, hi_closures=rows)
            dplan = device_plan(plan, cpu)
            f = CL.field(decomp)
            pad = decomp.n_halo_d; off = decomp.offset
            fill_range(d) = interface ? ((1 - pad[d]):(decomp.n_local[d] + pad[d])) :
                                        (1:decomp.n_local[d])
            for k in fill_range(3), j in fill_range(2), i in fill_range(1)
                gx, gy, gz = i + off[1], j + off[2], k + off[3]
                f[i+pad[1], j+pad[2], k+pad[3]] =
                    sin(0.31gx) + cos(0.17gy) + 0.5sin(0.23gz + 0.4gx)
            end
            CL.exchange_dim!(f, decomp, ax)
            out_h = CL.field(decomp); out_d = CL.field(decomp)
            CL.apply_along!(out_h, plan, f, decomp)
            CL.apply_along!(out_d, dplan, f, decomp)
            e = 0.0
            for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2],
                    i in 1:decomp.n_local[1]
                e = max(e, abs(out_h[i+pad[1], j+pad[2], k+pad[3]] -
                               out_d[i+pad[1], j+pad[2], k+pad[3]]))
            end
            check("$label device path bitwise, dim $ax split", gmax(e), 1e-300)
        end
    end
end

# ---------------------------------------------------------------------------
# Staged device exchange (reference/AMR_GPU.md). A device-resident field
# packs each halo slab into a contiguous backend stage by broadcast, crosses
# ranks through one contiguous copy each way around the same Sendrecv, and the
# fold pairing stages its whole blocks the same way. On host arrays those are
# the identical element copies, so FORCE_DEVICE_EXCHANGE pins the staged path
# bitwise here — combined with FORCE_KA and a DeviceBackend(CPU()) solver,
# a full decomposed run takes the closest CPU analog of a distributed device
# run: DevicePlans, kernel bodies, and staged communication together.
# ---------------------------------------------------------------------------
function test_staged_exchange()
    section("staged device exchange: decomposed runs bitwise")
    cpu = KernelAbstractions.CPU()
    function tube(backend)
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 0.6), (0.0, 0.3)), eos=eos,
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(SPLITN, 16, 12),
                     execution=Execution(dims=splitdims(1), storage=backend)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.05 * cos(2π * z / 0.3)))
        end)
        return s, Q
    end
    s1, Q1 = tube(CPUBackend())
    run!(s1, Q1; tfinal=0.02, nmax=6)
    CL.FORCE_KA[] = true
    CL.FORCE_DEVICE_EXCHANGE[] = true
    s2, Q2 = tube(DeviceBackend(cpu))
    try
        run!(s2, Q2; tfinal=0.02, nmax=6)
    finally
        CL.FORCE_KA[] = false
        CL.FORCE_DEVICE_EXCHANGE[] = false
    end
    check("staged multispecies tube, split x: bitwise",
          gmax(maximum(abs.(parent(Q1) .- parent(Q2)))), 1e-300)

    # Off-rank fold pairing: θ-split resolved axis, whole-block Sendrecv
    # through sendrecv_block!'s staging.
    function axis(backend)
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 0.5)),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                    metric=CylindricalMetric(), ic=nothing),
            Numerics(n_global=(20, SPLITN, 12),
                     execution=Execution(dims=splitdims(2), storage=backend)))
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, z) ->
            Prim(u=(0.05r * cos(θ), 0.2r, 0.05 * sin(2π * z / 0.5)),
                 p=1.0 + 0.02r^2, rho=1.0))
        return s, Q
    end
    s3, Q3 = axis(CPUBackend())
    run!(s3, Q3; tfinal=0.02, nmax=4)
    CL.FORCE_KA[] = true
    CL.FORCE_DEVICE_EXCHANGE[] = true
    s4, Q4 = axis(DeviceBackend(cpu))
    try
        run!(s4, Q4; tfinal=0.02, nmax=4)
    finally
        CL.FORCE_KA[] = false
        CL.FORCE_DEVICE_EXCHANGE[] = false
    end
    check("staged axis fold pair, split θ: bitwise",
          gmax(maximum(abs.(parent(Q3) .- parent(Q4)))), 1e-300)

    # The interface records of a patch layout and of a tiled level stage
    # through the backend the same way, the level transfer's chain runs on
    # the fine patches' device scratch, and the tag sweep evaluates on the
    # device; each is the host arithmetic element for element under the
    # toggles, so a rank-partitioned two-slab run and a decomposed tiled
    # regridding run are bitwise against the CPUBackend ones.
    function slabs(backend)
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                    transport=ConstantTransport(mu0=2e-2), ic=nothing),
            Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=false),
                     execution=Execution(patch_grid=(2, 1, 1), storage=backend)))
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) ->
            Prim(u=(0.5 + 0.1 * sin(2x), 0, 0), p=1.0 + 0.05 * cos(x),
                 rho=1.0 + 0.2 * sin(x)))
        run!(s, states; tfinal=0.3)
        return s, states
    end
    function tiled(backend)
        wall2 = (SlipWallBC(), SlipWallBC())
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=(wall2, per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(400, 1, 1), cfl=0.2,
                     patch_interfaces=PatchInterfaces(conservative=false),
                     execution=Execution(storage=backend),
                     amr=AMR(initial=BlockRegion((176, 0, 0), (16, 1, 1)), subcycle=true,
                             regrid_interval=5, tag_buffer=2, tile=8,
                             subdivide_lattice=false)))
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) -> x < 0.45 ?
            Prim(u=(0, 0, 0), p=1.0, rho=1.0) : Prim(u=(0, 0, 0), p=0.1, rho=0.125))
        run!(s, states; tfinal=0.03, nmax=41)
        return s, states
    end
    # A viscous level under `interface_flux = :ghost`, decomposed over the
    # ranks with the root: the gradient ring is taken through the fine
    # patch's device plans, and the packed rings cross the component-
    # distributed Allgatherv through the host staging.
    function ghost_level(backend)
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                    transport=ConstantTransport(mu0=2e-2), ic=nothing),
            Numerics(n_global=(192, 1, 1), art=ArtificialProperties(enabled=false),
                     patch_interfaces=PatchInterfaces(flux=:ghost, conservative=false),
                     execution=Execution(storage=backend),
                     amr=AMR(initial=BlockRegion((80, 0, 0), (32, 1, 1)), regrid_interval=0)))
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) ->
            Prim(u=(0.5 + 0.1 * sin(2x), 0, 0), p=1.0 + 0.05 * cos(x),
                 rho=1.0 + 0.2 * sin(x)))
        run!(s, states; tfinal=0.3, nmax=6)
        return s, states
    end
    # A tiled level along the plane z = 0 of an r-z run from the axis
    # outward, decomposed with the root: the corner tile folds on both
    # dimensions and its neighbours on z alone, each fold's mirror fill on
    # the tile's edge rank and on every member of a stack, and the box
    # mirror across both folds on every owner.
    function folded_level(backend)
        s = build_solver(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                 bcs=((AxisBC(), SlipWallBC()), per3[2],
                                 (SymmetryPlaneBC(), SlipWallBC())),
                                 metric=CylindricalMetric(),
                                 transport=ConstantTransport(mu0=1e-3), ic=nothing),
                         Numerics(n_global=(48, 1, 48), cfl=0.5,
                                  art=ArtificialProperties(enabled=false),
                                  filter=StateFilter(interval=1),
                                  patch_interfaces=PatchInterfaces(conservative=false),
                                  execution=Execution(storage=backend),
                                  amr=AMR(initial=BlockRegion((0, 0, 0), (25, 1, 9)),
                                          regrid_interval=0, tile=8)))
        states = allocate_state(s)
        initialize!(s, states, (r, θ, z) -> begin
            rho = 1 + 0.05 * exp(-((sqrt(r^2 + z^2) - 0.4) / 0.15)^2)
            Prim(rho=rho, u=(0.02 * r, 0.0, 0.01 * z), p=rho^1.4)
        end)
        run!(s, states; tfinal=0.05, nmax=4)
        return s, states
    end
    # The conservative coarse-fine coupling is host-only and leaves a stacked
    # device level uncorrected (src/amr/reflux.jl), so the refined builds above
    # pass `interface_conservative = false` and both runs go without it.
    for (label, build) in (("two viscous slabs", slabs),
                           ("tiled regridding Sod", tiled),
                           ("ghost-flux viscous level", ghost_level),
                           ("tiled level at the r-z folds", folded_level))
        s5, q5 = build(CPUBackend())
        s6, q6 = CL.with_knob(CL.FORCE_KA => true, CL.FORCE_DEVICE_EXCHANGE => true) do
            build(DeviceBackend(cpu))
        end
        d = 0.0
        for i in eachindex(q5.patches)
            d = max(d, maximum(abs.(parent(q5.patches[i]) .- parent(q6.patches[i]))))
        end
        check("staged $label: patch count matches", abs(length(q5.patches) - length(q6.patches)), 0.5)
        check("staged $label: step count matches", abs(s5.step - s6.step), 0.5)
        check("staged $label: bitwise", gmax(d), 1e-300)
    end
end
