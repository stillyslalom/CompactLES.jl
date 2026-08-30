# Precompile workload.
#
# The cost this addresses is that every distinct `Solver{...}` type forces a
# fresh compilation of the call tree beneath `step!` and `compute_rhs!`, and
# that a multi-rank test run pays it independently on every rank: 72 s per
# rank at np = 2 on the workstation, 183 s on the CI runner, and 628 s at
# np = 8 there, where ranks spinning in a collective take the cores from
# ranks still compiling. The workload below builds the solver configurations
# the test suites use, on small grids at np = 1, and runs each through the
# entry points the tests call, so those trees are compiled once into the
# package image. Measured: per-rank compile of test/mpi_tests.jl at np = 2
# fell from 72 s to 17 s and the wall time from 99 s to 38 s, against 55 s
# added to the package precompile.
#
# A `Decomp` needs an initialized communicator, so the workload initializes
# MPI as a singleton inside the precompile process. That is fine with the
# bundled binaries, and it is skipped when `MPIPreferences.binary` is
# `"system"`: a singleton init on a cluster login node may fail or hang under
# the scheduler's process manager, and installing the package must not depend
# on it. Whether it can be enabled there is an open item in
# reference/CLUSTER.md.
#
# Two limits. Closures are typed by their defining module, so a test script's
# initial-condition function still compiles `initialize!` for itself; and the
# split-dimension paths (reduced-interface solves, off-rank folds) execute only
# at np > 1, which no workload reaches. Both are in the residue above.

using PrecompileTools
using MPIPreferences: MPIPreferences

if MPIPreferences.binary != "system"
@setup_workload begin
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    per = (PeriodicBC(), PeriodicBC())
    per3 = (per, per, per)
    walls = (SlipWallBC(), SlipWallBC())
    off = ArtParams(enabled=false)
    cpu = KernelAbstractions.CPU()
    @compile_workload begin
        # Periodic and closed C6 / C10 derivatives.
        for deriv in (lele_d1_6(), lele_d1_10()), bcs in (per3, (walls, per, per))
            s = Solver(n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0), bcs=bcs,
                       deriv=deriv, art=off)
            f = field(s.decomp); df = field(s.decomp)
            f .= 1.0
            exchange_halos!(f, s.decomp)
            for ax in 1:3
                deriv_along!(df, f, s, ax, 1); _scale_grad!(df, s, ax)
            end
        end
        # Every metric and fold, through the RHS.
        cases = (
            (; n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
               bcs=per3, kw=(;)),
            (; n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
               bcs=(walls, per, per),
               kw=(; stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing, nothing))),
            (; n_global=(12, 16, 12), L_domain=(1.0, 2π, 0.5), metric=CylindricalMetric(),
               bcs=(walls, per, per), kw=(; origin=(0.2, 0.0, 0.0))),
            (; n_global=(16, 16, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per, per), kw=(;)),
            (; n_global=(16, 16, 12), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
               bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per), kw=(;)),
        )
        for cs in cases
            kw = merge((; n_global=cs.n_global, L_domain=cs.L_domain, metric=cs.metric,
                        bcs=cs.bcs, art=off), cs.kw)
            s = Solver(; kw...)
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
            apply_bcs!(s, Q)
            compute_rhs!(s, Q, zero(Q))
        end
        # Artificial properties: every β sensor and both detectors.
        for art in (ArtParams(enabled=true, beta_sensor=:strain),
                    ArtParams(enabled=true, beta_sensor=:gated_strain),
                    ArtParams(enabled=true, beta_sensor=:dilatation),
                    ArtParams(enabled=true, beta_sensor=:ungated_dilatation),
                    ArtParams(enabled=true, detector=:d8))
            s = Solver(n_global=(12, 12, 12), L_domain=(2π, 2π, 2π), bcs=per3, art=art)
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) ->
                Prim(u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.3sin(2z)),
                     p=1 + 0.2cos(x), rho=1 + 0.3sin(y)))
            compute_rhs!(s, Q, zero(Q))
        end
        # Timestepping through run!: periodic, outflow, and a switchable outflow.
        for xbc in (per, (SlipWallBC(), NSCBCOutflowBC(pinf=1.0)),
                    (SlipWallBC(), SwitchableBC(SlipWallBC(), NSCBCOutflowBC(pinf=1.0))))
            s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                                 bcs=(xbc, per, per),
                                 ic=(x, y, z) -> Prim(u=(0, 0, 0),
                                                      p=1 + 4exp(-200(x - 0.3)^2),
                                                      rho=1 + exp(-200(x - 0.3)^2))),
                         Numerics(n_global=(16, 12, 12)))
            run!(s, Q; tfinal=1e9, nmax=2)
            compute_dt(s, Q)
        end
        # Two species on the host and device backends, Cartesian and cylindrical.
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        for backend in (CPUBackend(), DeviceBackend(cpu))
            s = Solver(n_global=(16, 12, 12), L_domain=(1.0, 0.6, 0.3), eos=eos,
                       bcs=(walls, per, per), backend=backend)
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) -> begin
                θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
                Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                     u=(0.1 * sin(2π * y / 0.6), 0.0, 0.05 * cos(2π * z / 0.3)))
            end)
            run!(s, Q; tfinal=0.02, nmax=2)
            positivity_floors(s, Q, StepControl(floor_ratio=1e-6))
            s = Solver(n_global=(12, 16, 12), L_domain=(1.0, 2π, 0.5),
                       bcs=((AxisBC(), SlipWallBC()), per, per),
                       metric=CylindricalMetric(), backend=backend)
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
        s = Solver(n_global=(48, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3, art=off,
                   filter_interval=0, patch_grid=(2, 1, 1))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(u=(0.5, 0, 0), p=1.0, rho=1.0 + 0.2sin(x)))
        run!(s, Q; tfinal=0.5, nmax=2)
        r1 = BlockRegion((16, 0, 0), (16, 1, 1))
        r2 = BlockRegion((3 * 16 + 10, 0, 0), (20, 1, 1))
        for subcycle in (false, true), refine in (r1, [r1, r2])
            s = Solver(n_global=(48, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3, art=off,
                       filter_interval=0, subcycle=subcycle, refine=refine)
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) -> Prim(u=(0.5, 0, 0), p=1.0, rho=1.0 + 0.2sin(x)))
            run!(s, Q; tfinal=0.5, nmax=2)
            for (ps, q) in eachpatch(s, Q)
                xcoord(ps, 1, 1)
            end
        end
        for subcycle in (false, true)
            s = Solver(n_global=(48, 48, 1), L_domain=(2π, 2π, 1.0), bcs=per3, art=off,
                       filter_interval=0, subcycle=subcycle, tile=12,
                       refine=BlockRegion((12, 12, 0), (24, 24, 1)))
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) ->
                Prim(u=(0.4, 0.3, 0), p=1.0, rho=1.0 + 0.1sin(x)sin(y)))
            run!(s, Q; tfinal=0.5, nmax=1)
            _sync_level_records!(s, Q, s.levels[2])
        end
        s = Solver(n_global=(120, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(walls, per, per), cfl=0.2,
                   refine=BlockRegion((40, 0, 0), (30, 1, 1)),
                   subcycle=true, regrid_interval=2, tag_buffer=8)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> x < 0.45 ?
            Prim(u=(0, 0, 0), p=1.0, rho=1.0) : Prim(u=(0, 0, 0), p=0.1, rho=0.125))
        run!(s, Q; tfinal=0.03, nmax=3)
        refined_region(s)
    end
end
end
