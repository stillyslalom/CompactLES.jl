# The test suites' precompile workload: the solver configurations the serial
# and multi-rank suites build, on small grids at np = 1, each run through the
# entry points the tests call. The helper package test/TestPrecompile includes
# this file and runs it while it precompiles, which holds the constants its
# blocks are gated on; CompactLES's own image holds none of it.
#
# Compiled code is keyed on the `Solver` type, not on the decomposition. The
# split-dimension paths (reduced-interface solves, off-rank folds, halo
# exchange with a neighbour) are compiled with the tree that contains them
# whether or not a run at np = 1 executes them, and a `--trace-compile` of two
# phases of the MPI suite at np = 2 finds no CompactLES method compiled for the
# decomposition alone. What a rank compiles is the `Solver` types the image
# lacks. That type carries the precision, the metric, the equation of state
# and transport types, the fold of each face and which dimensions are
# active, so a fold on the second dimension, or a third active dimension, is
# a separate tree. Measured with these configurations in the image: per-rank
# compile of test/mpi_tests.jl at np = 2 fell from 72 s to 17 s and the wall
# time from 99 s to 38 s.
#
# The configurations are built as the suites build theirs (build_solver.jl),
# so that each reads as its test does.

include(joinpath(@__DIR__, "build_solver.jl"))

# The configurations of the serial suite and of the multi-rank suite's
# ordinary phases.
function _test_suite_workload(cpu)
    per = (PeriodicBC(), PeriodicBC())
    per3 = (per, per, per)
    walls = (SlipWallBC(), SlipWallBC())
    off = ArtificialProperties(enabled=false)
    # Periodic and closed C6 / C10 derivatives.
    for deriv in (lele_d1_6(), lele_d1_10()), bcs in (per3, (walls, per, per))
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs, ic=nothing),
            Numerics(n_global=(16, 12, 12), deriv=deriv, art=off))
        f = field(s.decomp); df = field(s.decomp)
        f .= 1.0
        exchange_halos!(f, s.decomp)
        for ax in 1:3
            deriv_along!(df, f, s, ax, 1); _scale_grad!(df, s, ax)
        end
    end
    # Every metric and fold, through the RHS.
    unit = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    none = (nothing, nothing, nothing)
    cases = (
        (; n_global=(16, 12, 12), domain=unit, metric=CartesianMetric(), bcs=per3,
           stretch=none),
        (; n_global=(16, 12, 12), domain=unit, metric=CartesianMetric(),
           bcs=(walls, per, per),
           stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing, nothing)),
        (; n_global=(12, 16, 12), domain=((0.2, 1.2), (0.0, 2π), (0.0, 0.5)),
           metric=CylindricalMetric(), bcs=(walls, per, per), stretch=none),
        (; n_global=(16, 16, 1), domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)),
           metric=CylindricalMetric(), bcs=((AxisBC(), SlipWallBC()), per, per),
           stretch=none),
        (; n_global=(16, 16, 12), domain=((0.0, 1.0), (0.0, π), (0.0, 2π)),
           metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per), stretch=none),
    )
    for cs in cases
        s = build_solver(Problem(; domain=cs.domain, metric=cs.metric, bcs=cs.bcs,
                                 ic=nothing),
                         Numerics(; n_global=cs.n_global, stretch=cs.stretch, art=off))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(s, Q)
        compute_rhs!(s, Q, zero(Q))
    end
    # Artificial properties: every β sensor and both detectors.
    for art in (ArtificialProperties(enabled=true, beta_sensor=:strain),
                ArtificialProperties(enabled=true, beta_sensor=:gated_strain),
                ArtificialProperties(enabled=true, beta_sensor=:dilatation),
                ArtificialProperties(enabled=true, beta_sensor=:ungated_dilatation),
                ArtificialProperties(enabled=true, detector=:d8))
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
            Numerics(n_global=(12, 12, 12), art=art))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) ->
            Prim(u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.3sin(2z)),
                 p=1 + 0.2cos(x), rho=1 + 0.3sin(y)))
        compute_rhs!(s, Q, zero(Q))
    end
    # Two species, Cartesian and cylindrical, on the host backend and,
    # under `PRECOMPILE_DEVICE`, on the device backend.
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)])
    for backend in (PRECOMPILE_DEVICE ? (CPUBackend(), DeviceBackend(cpu)) :
                    (CPUBackend(),))
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 0.6), (0.0, 0.3)), eos=eos,
                    bcs=(walls, per, per), ic=nothing),
            Numerics(n_global=(16, 12, 12), execution=Execution(storage=backend)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.05 * cos(2π * z / 0.3)))
        end)
        run!(s, Q; tfinal=0.02, nmax=2)
        positivity_floors(s, Q, StepControl(floor_ratio=1e-6))
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 0.5)),
                    bcs=((AxisBC(), SlipWallBC()), per, per), metric=CylindricalMetric(),
                    ic=nothing),
            Numerics(n_global=(12, 16, 12), execution=Execution(storage=backend)))
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, z) ->
            Prim(u=(0.05r * cos(θ), 0.2r, 0.05 * sin(2π * z / 0.5)),
                 p=1.0 + 0.02r^2, rho=1.0))
        run!(s, Q; tfinal=0.02, nmax=2)
    end
    # Device plans against their host plans.
    for (scheme, periodic) in ((lele_d1_6(Float64), true), (lele_d1_10(Float64), false))
        decomp = Decomp((16, 12, 12), (periodic, periodic, periodic))
        plan = plan_direction(decomp, scheme, 1, 0.05)
        dplan = device_plan(plan, cpu)
        f = field(decomp); f .= 1.0
        out = field(decomp)
        apply_along!(out, plan, f, decomp)
        apply_along!(out, dplan, f, decomp)
    end
    # Patches and levels: two patches, two and three levels, tiles, a regrid.
    s = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(48, 1, 1), art=off, filter=StateFilter(interval=0),
                 execution=Execution(patch_grid=(2, 1, 1))))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0.5, 0, 0), p=1.0, rho=1.0 + 0.2sin(x)))
    run!(s, Q; tfinal=0.5, nmax=2)
    r1 = BlockRegion((16, 0, 0), (16, 1, 1))
    r2 = BlockRegion((3 * 16 + 10, 0, 0), (20, 1, 1))
    for subcycle in (false, true), refine in (r1, [r1, r2])
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(48, 1, 1), art=off, filter=StateFilter(interval=0),
                     amr=AMR(initial=refine, regrid_interval=0, subcycle=subcycle)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(u=(0.5, 0, 0), p=1.0, rho=1.0 + 0.2sin(x)))
        run!(s, Q; tfinal=0.5, nmax=2)
        for (ps, q) in eachpatch(s, Q)
            xcoord(ps, 1, 1)
        end
    end
    for subcycle in (false, true)
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(48, 48, 1), art=off, filter=StateFilter(interval=0),
                     amr=AMR(initial=BlockRegion((12, 12, 0), (24, 24, 1)),
                             regrid_interval=0, subcycle=subcycle, tile=12)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) ->
            Prim(u=(0.4, 0.3, 0), p=1.0, rho=1.0 + 0.1sin(x)sin(y)))
        run!(s, Q; tfinal=0.5, nmax=1)
        _sync_level_records!(s, Q, s.levels[2])
    end
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, per, per),
                ic=nothing),
        Numerics(n_global=(120, 1, 1), cfl=0.2,
                 amr=AMR(initial=BlockRegion((40, 0, 0), (30, 1, 1)), subcycle=true,
                         regrid_interval=2, tag_buffer=8)))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> x < 0.45 ?
        Prim(u=(0, 0, 0), p=1.0, rho=1.0) : Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    run!(s, Q; tfinal=0.03, nmax=3)
    refined_region(s)
    # Float32 throughout, under `PRECOMPILE_FLOAT32`. A second element type
    # recompiles the whole tree beneath step!, and the suites reach it on a
    # plain grid, on a refined hierarchy and, under `PRECOMPILE_DEVICE`, on
    # the device backend. The saving lands only on the Solver type tuples
    # named here; one that is not compiles in full wherever it is first
    # built, which is why the 3-D refined device case is left to the suite
    # rather than spending a third dimension of image on it.
    if PRECOMPILE_FLOAT32
        T = Float32
        for storage in (PRECOMPILE_DEVICE ? (T, DeviceBackend(cpu) => T) : (T,))
            s = build_solver(Problem(domain=((0.0, one(T)), (0.0, one(T)), (0.0, one(T))),
                                     bcs=per3, ic=nothing),
                             Numerics(n_global=(16, 12, 12), cfl=T(0.2),
                                      art=ArtificialProperties(enabled=false),
                                      execution=Execution(storage=storage)))
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) ->
                Prim(rho=1 + T(0.01) * sin(2π * x), T_ion=one(T),
                     u=(T(0.1), zero(T), zero(T))))
            run!(s, Q; tfinal=T(0.01), nmax=2)
            compute_dt(s, Q)
        end
        s = build_solver(Problem(domain=((0.0, T(2π)), (0.0, one(T)), (0.0, one(T))),
                                 bcs=per3, ic=nothing),
                         Numerics(n_global=(48, 1, 1), cfl=T(0.2),
                                  art=ArtificialProperties(enabled=false),
                                  filter=StateFilter(interval=0),
                                  execution=Execution(storage=T),
                                  amr=AMR(initial=BlockRegion((20, 0, 0), (8, 1, 1)),
                                          regrid_interval=0)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) ->
            Prim(u=(T(0.5), zero(T), zero(T)), p=one(T),
                 rho=1 + T(0.1) * sin(x)))
        run!(s, Q; tfinal=T(0.01), nmax=2)
    end
    # The device backend on a subcycled, tiled, regridding hierarchy, with
    # the device-storage branches and the KA bodies both selected. On a real
    # device those are the only path; the suite reaches them through the two
    # toggles, which is why they are set here rather than left at default.
    # Behind `PRECOMPILE_DEVICE`, and so off unless a checkout asks for it.
    if PRECOMPILE_DEVICE
        with_knob(FORCE_KA => true, FORCE_DEVICE_EXCHANGE => true) do
            s = build_solver(
                Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, per, per),
                        ic=nothing),
                Numerics(n_global=(201, 1, 1), cfl=0.2,
                         execution=Execution(storage=DeviceBackend(cpu)),
                         amr=AMR(initial=BlockRegion((85, 0, 0), (31, 1, 1)), subcycle=true,
                                 regrid_interval=5, tile=8)))
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) -> x < 0.5 ?
                Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                Prim(u=(0, 0, 0), p=0.1, rho=0.125))
            run!(s, Q; tfinal=0.06, nmax=3)
            # The same bodies on host storage, the form FORCE_KA selects.
            s = build_solver(
                Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
                Numerics(n_global=(16, 12, 12)))
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) ->
                Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0))
            run!(s, Q; tfinal=0.02, nmax=2)
        end
    end
    return nothing
end

# The solver configurations of the compile-bound eight-rank phases of
# `test/mpi_tests.jl`, each built as its phase builds it, with the keywords in
# the phase's order and of its types, at np = 1. A grid or a step count may be
# smaller than the phase's where that leaves the types and the paths taken
# unchanged.
function _mpi_suite_workload(cpu)
    per = (PeriodicBC(), PeriodicBC())
    per3 = (per, per, per)
    walls = (SlipWallBC(), SlipWallBC())
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    noart = ArtificialProperties(enabled=false)
    world = MPI.COMM_WORLD
    whole = (1, 1, 1)
    # "symmetry plane": two species, constant viscosity, the artificial
    # properties and a filter every step, with the planes on one dimension at
    # a time in both precisions, on the first two together, and on the high
    # end of a line whose low end is a Dirichlet inflow.
    function plane_case(T, ng, bcs, ic, detector, channel)
        local s, Q
        s = build_solver(Problem(domain=((0.0, one(T)), (0.0, one(T)), (0.0, one(T))),
                                 bcs=bcs,
                                 eos=IdealMixture([IdealSpecies{T}("a", T(1), T(1.4)),
                                              IdealSpecies{T}("b", T(0.7), T(1.3))]),
                                 transport=ConstantTransport{T}(mu0=T(0.01)), ic=nothing),
                         Numerics(n_global=ng,
                                  art=ArtificialProperties{T}(enabled=true, detector=detector,
                                                         species_flux=channel),
                                  deriv=lele_d1_6(T), cfl=T(0.4),
                                  filter=StateFilter(compact_filter(T(0.45), T), interval=1,
                                                     cfl=zero(T)),
                                  execution=Execution(comm=world, dims=whole)))
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        apply_bcs!(s, Q)
        compute_rhs!(s, Q, zero(Q))
        run!(s, Q; tfinal=1e9, nmax=2)
    end
    for T in (Float64, Float32), ax in 1:3
        ic = (x, y, z) -> begin
            r = (x, y, z)[ax]
            Prim(rho=one(T) + T(0.05) * cos(T(pi) * r), T_ion=T(2), Y=(T(0.4), T(0.6)),
                 u=ntuple(d -> d == ax ? T(0.1) * sin(T(pi) * r) : zero(T), 3))
        end
        plane_case(T, ntuple(d -> d == ax ? 16 : 1, 3),
                   ntuple(d -> d == ax ? sym : per, 3), ic,
                   ax == 3 ? :d8 : :delta4, (:partial_density, :bulk, :fickian)[ax])
    end
    flat = (x, y, z) -> Prim(rho=1.0 + 0.05x, T_ion=2.0, Y=(0.4, 0.6), u=(0.1, 0.0, 0.0))
    plane_case(Float64, (16, 12, 1), (sym, sym, per), flat, :delta4, :fickian)
    inflow = Prim(rho=1.0, T_ion=2.0, Y=(0.4, 0.6), u=(-0.2, 0.0, 0.0))
    plane_case(Float64, (16, 1, 1),
               ((DirichletBC((x, y, z, t) -> inflow), SymmetryPlaneBC()), per, per),
               flat, :delta4, :fickian)
    # "level at a symmetry plane" and "level on the axis": two species, the
    # artificial properties under the `:d8` detector and a refined level
    # reaching the fold, as one box and as tiles.
    pair = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                         IdealSpecies{Float64}("b", 0.7, 1.3)])
    mixed = (x, y, z) -> Prim(rho=1 + 0.1 * cos(pi * x), u=(0.05 * sin(pi * x), 0.0, 0.0),
                              p=1.0, Y=(0.5 + 0.1 * cos(pi * x), 0.5 - 0.1 * cos(pi * x)))
    for tile in (0, 5)
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(sym, per, per),
                    eos=pair, transport=ConstantTransport{Float64}(mu0=0.01), ic=nothing),
            Numerics(n_global=(48, 24, 1), cfl=0.4,
                     art=ArtificialProperties(enabled=true, detector=:d8),
                     filter=StateFilter(interval=1), execution=Execution(comm=world),
                     amr=AMR(initial=BlockRegion((0, 4, 0), (10, 16, 1)), regrid_interval=0,
                             tile=tile)))
        Q = allocate_state(s)
        initialize!(s, Q, mixed)
        run!(s, Q; tfinal=1e9, nmax=2)
        domain_volume(s)
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((AxisBC(), SlipWallBC()), per, (SymmetryPlaneBC(), SlipWallBC())),
                    metric=CylindricalMetric(), eos=pair,
                    transport=ConstantTransport{Float64}(mu0=0.01), ic=nothing),
            Numerics(n_global=(48, 1, 36), cfl=0.4,
                     art=ArtificialProperties(enabled=true, detector=:d8),
                     filter=StateFilter(interval=1), execution=Execution(comm=world),
                     amr=AMR(initial=BlockRegion((0, 0, 0), (10, 1, 10)), regrid_interval=0,
                             tile=tile)))
        Q = allocate_state(s)
        initialize!(s, Q, mixed)
        run!(s, Q; tfinal=1e9, nmax=2)
        domain_volume(s)
    end
    # "staggered operators": the operator on a bare decomposition, periodic,
    # closed and folded under both parities, and from solvers on a stretched
    # line, the resolved axis and the spherical origin and poles; then the
    # implicit stage on walls of both parities, a periodic box and the axis.
    for ax in 1:3, (periodic, parity, folded) in ((true, 1, false), (false, 1, false),
                                                  (false, -1, false), (false, 1, true),
                                                  (false, -1, true))
        decomp = Decomp(ntuple(d -> d == ax ? 16 : 6, 3),
                        ntuple(d -> d == ax ? periodic : true, 3); comm=MPI.COMM_SELF)
        σ = folded ? parity : nothing
        op = StaggeredDiffusion(decomp, ax, 1 / 16; parity=parity, lo_fold=σ, hi_fold=σ)
        T = field(decomp); κ = field(decomp); out = field(decomp)
        fill!(T, 1.0); fill!(κ, 1.0)
        staggered_diffusion!(out, op, T, κ, decomp)
    end
    stretched = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, per, per),
                ic=nothing),
        Numerics(n_global=(16, 12, 1),
                 stretch=(sine_cluster(0.0, 1.0, 0.3, 0.4), nothing, nothing), art=noart,
                 execution=Execution(comm=world, dims=whole)))
    cyl(n) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per, per), ic=nothing),
        Numerics(n_global=(n, 16, 1), art=noart,
                 execution=Execution(comm=world, dims=whole)))
    sph = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)), metric=SphericalMetric(),
                bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per), ic=nothing),
        Numerics(n_global=(16, 16, 12), art=noart,
                 execution=Execution(comm=world, dims=whole)))
    for (s, d) in ((stretched, 1), (cyl(16), 1), (sph, 1), (sph, 2))
        op = StaggeredDiffusion(s, d)
        T = field(s.decomp); κ = field(s.decomp); L = field(s.decomp)
        fill!(T, 1.0); fill!(κ, 1.0)
        staggered_diffusion!(L, op, T, κ, s.decomp)
    end
    closed = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, per, per),
                ic=nothing),
        Numerics(n_global=(16, 16, 1), art=noart,
                 execution=Execution(comm=world, dims=whole)))
    box = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(16, 16, 1), art=noart,
                 execution=Execution(comm=world, dims=whole)))
    for (s, parity) in ((closed, 1), (closed, -1), (box, 1), (cyl(12), 1))
        stage = DiffusionStage(s; parity=parity)
        r = field(s.decomp); κ = field(s.decomp); out = field(s.decomp)
        fill!(κ, 1.0)
        for I in interior(s.decomp)
            r[I] = 1 + 0.1 * I[1]
        end
        T = copy(r)
        solve_stage!(T, stage, r, κ, 0.1)
        stage_operator!(out, stage, T, κ, 0.1)
    end
    # "implicit conduction": the additive pair on a periodic box under the
    # default step rule, and on a line under `:none` with rollbacks.
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                transport=ConstantTransport(mu0=1e-3, Pr=1e-3 * 3.5 / 2.5, Sc=1.0),
                ic=nothing),
        Numerics(n_global=(24, 36, 1), art=noart, cfl=0.5,
                 implicit=ImplicitConduction(rtol=1e-12),
                 execution=Execution(comm=world, dims=whole)))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(rho=1 / (1 + 0.3exp(-((x - 0.4)^2 +
                                                              (y - 0.55)^2) / 0.02)),
                                        u=(0.2sinpi(2y), 0.2cospi(2x), 0.0), p=1.0))
    run!(s, Q; tfinal=1e9, nmax=2)
    n = 72
    h = 2 / n
    gas = 1e-10
    mu = 1e-3 * h^2
    s = build_solver(
        Problem(domain=((-1.0, -1.0 + 2.0), (0.0, 0.0 + 1.0), (0.0, 0.0 + 1.0)), bcs=per3,
                eos=IdealSpecies("gas"; R=gas, gamma=1 + gas),
                transport=ConstantTransport(mu0=mu, Pr=mu * (1 + gas), Sc=1.0), ic=nothing),
        Numerics(n_global=(n, 1, 1), art=noart, cfl=1e6, control=StepControl(retries=6),
                 implicit=ImplicitConduction(step_rule=:none, rtol=1e-12),
                 filter=StateFilter(interval=0),
                 execution=Execution(comm=world, dims=whole)))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                        p=gas * (abs(x + 0.5) < h / 2 ? 1.2 : 0.2)))
    # The spike's first attempts fail and roll back, as the phase's do; the
    # warnings each rollback raises are kept out of the precompile output.
    Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        run!(s, Q; tfinal=32 * h^2 / 2)
    end
    # "positivity limiter": a blast in a box, through the spherical origin, at
    # the corner of the r-z axis and a plane, on two same-level patches, on
    # refined tiles at the global step and subcycled, Cartesian and r-z, and
    # the shell's fallback on a subcycled box; under `PRECOMPILE_DEVICE` the
    # box and the patches on device storage as well. The patches and tiles run
    # under either interface flux. A limited solver keeps `:ghost` only under
    # `interface_limited_ghost`, as the phase passes it; without it the
    # constructor takes `:closure` and warns, so a `:ghost` case would take
    # the closure path a second time and the ghost path would be compiled at
    # run time on every rank.
    gas = IdealSpecies("gas"; gamma=1.4, R=1.0)
    permissive = StepControl(validity=:permissive)
    blast = (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                              p=1e-3 + exp(-((x - 0.45)^2 + (y - 0.5)^2) / 0.004))
    corner = (r, θ, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                               p=1e-4 + exp(-(r^2 + z^2) / 0.06^2))
    function limited(s, ic, nmax)
        local Q = allocate_state(s)
        initialize!(s, Q, ic)
        run!(s, Q; tfinal=1.0, nmax=nmax)
        positivity_counts(s)
        state_report(s, Q)
    end
    limited(build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, per, per), eos=gas,
                ic=nothing),
        Numerics(n_global=(36, 36, 1), art=ArtificialProperties(enabled=true), cfl=0.4,
                 control=permissive, positivity_limiter=true,
                 execution=Execution(comm=world, dims=whole))), blast, 20)
    limited(build_solver(
        Problem(domain=((0.0, 0.0 + 1.0), (π / 2, π / 2 + 1.0), (0.0, 0.0 + 1.0)),
                metric=SphericalMetric(), bcs=((OriginBC(), SlipWallBC()), per, per),
                eos=gas, ic=nothing),
        Numerics(n_global=(48, 1, 1), art=ArtificialProperties(enabled=true), cfl=0.3,
                 control=permissive, positivity_limiter=true,
                 execution=Execution(comm=world, dims=whole))),
            (r, θ, φ) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1e-4 + exp(-(r / 0.08)^2)),
            30)
    limited(build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per, (SymmetryPlaneBC(), SlipWallBC())),
                eos=gas, ic=nothing),
        Numerics(n_global=(36, 1, 36), art=ArtificialProperties(enabled=true), cfl=0.3,
                 control=permissive, positivity_limiter=true,
                 execution=Execution(comm=world, dims=whole))), corner, 20)
    for flux in (:closure, :ghost)
        limited(build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, per, per),
                    eos=gas, ic=nothing),
            Numerics(n_global=(96, 32, 1), art=ArtificialProperties(enabled=true), cfl=0.4,
                     control=permissive, positivity_limiter=true,
                     patch_interfaces=PatchInterfaces(flux=flux, limited_ghost=true),
                     execution=Execution(patch_grid=(2, 1, 1), comm=world))), blast, 20)
    end
    for flux in (:closure, :ghost), sub in (false, true)
        limited(build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, walls, per),
                    eos=gas, ic=nothing),
            Numerics(n_global=(48, 48, 1), art=ArtificialProperties(enabled=true), cfl=0.4,
                     control=permissive, positivity_limiter=true,
                     patch_interfaces=PatchInterfaces(flux=flux, limited_ghost=true),
                     execution=Execution(comm=world),
                     amr=AMR(initial=BlockRegion((12, 12, 0), (24, 24, 1)),
                             regrid_interval=0, subcycle=sub, tile=8))), blast, 10)
        limited(build_solver(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                     metric=CylindricalMetric(),
                                     bcs=((AxisBC(), SlipWallBC()), per,
                                     (SymmetryPlaneBC(), SlipWallBC())),
                                     eos=gas, ic=nothing),
                             Numerics(n_global=(36, 1, 36),
                                      art=ArtificialProperties(enabled=true), cfl=0.3,
                                      control=permissive, positivity_limiter=true,
                                      patch_interfaces=PatchInterfaces(flux=flux,
                                                                       limited_ghost=true),
                                      execution=Execution(comm=world),
                                      amr=AMR(initial=BlockRegion((0, 0, 0), (17, 1, 17)),
                                              regrid_interval=0, subcycle=sub, tile=8))),
                corner, 10)
    end
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, walls, per),
                eos=gas, transport=ConstantTransport(mu0=0.0), ic=nothing),
        Numerics(n_global=(48, 48, 1), art=ArtificialProperties(enabled=true), cfl=0.4,
                 control=permissive, execution=Execution(comm=world),
                 amr=AMR(initial=BlockRegion((20, 18, 0), (10, 10, 1)), regrid_interval=0,
                         subcycle=true)))
    Q = allocate_state(s)
    initialize!(s, Q, blast)
    run!(s, Q; tfinal=1.0, nmax=10)
    # The same box split and the same patches on device storage under the two
    # toggles, and on the host without them, as the phase runs them.
    for device in (PRECOMPILE_DEVICE ? (false, true) : (false,))
        backend = device ? DeviceBackend(cpu) : CPUBackend()
        with_knob(FORCE_KA => device, FORCE_DEVICE_EXCHANGE => device) do
            s = build_solver(
                Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, per, per),
                        eos=gas, ic=nothing),
                Numerics(n_global=(36, 36, 1), art=ArtificialProperties(enabled=true),
                         cfl=0.4, control=permissive, positivity_limiter=true,
                         execution=Execution(dims=whole, storage=backend)))
            Q = allocate_state(s)
            initialize!(s, Q, blast)
            run!(s, Q; tfinal=1.0, nmax=10)
            for flux in (:closure, :ghost)
                s = build_solver(
                    Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                            bcs=(walls, per, per), eos=gas, ic=nothing),
                    Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=true),
                             cfl=0.3, control=permissive, positivity_limiter=true,
                             filter=StateFilter(cfl=0.35),
                             patch_interfaces=PatchInterfaces(flux=flux, limited_ghost=true),
                             execution=Execution(patch_grid=(2, 1, 1), storage=backend)))
                Q = allocate_state(s)
                initialize!(s, Q, (x, y, z) ->
                    Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                         p=1000 * (1 - tanh_blend(x, 0.45, 2 / 95)) + 0.01))
                run!(s, Q; tfinal=1.0, nmax=10)
            end
        end
    end
    return nothing
end
