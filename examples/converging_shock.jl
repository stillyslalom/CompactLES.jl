# Cylindrically converging shock in an effectively one-dimensional radial mesh.
#
# Run: julia --project=. examples/converging_shock.jl nx=256 tfinal=0.1 nmax=200
#      mpiexec -n 2 julia --project=. -t 1 examples/converging_shock.jl nx=512

using CompactLES
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

const opt = CompactLES.script_args(ARGS, (nx=1024, tfinal=0.35, nmax=1_000_000,
                               every=100, output=""); positional=(:nx, :tfinal))

mpi_main() do
    problem = Problem(
        name="converging shock",
        eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
        metric=CylindricalMetric(),
        domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        bcs=((AxisBC(), SlipWallBC()), PeriodicBC(), PeriodicBC()),
        ic=(r, theta, z, h) -> begin
            drive = tanh_blend(r, 0.7, 3h)
            Prim(rho=1.0 + 3.0 * drive, p=1.0 + 19.0 * drive)
        end,
    )
    numerics = Numerics(n_global=(opt.nx, 1, 1), art=ArtParams(enabled=true),
                        cfl=0.4, filter_interval=1,
                        dims=(MPI.Comm_size(MPI.COMM_WORLD), 1, 1))
    solver, Q = setup(problem, numerics)
    run!(solver, Q; tfinal=opt.tfinal, nmax=opt.nmax,
         callback=ProgressLog(every=opt.every, tfinal=opt.tfinal))

    r, rho = line_profile(solver, Q, :rho)
    if MPI.Comm_rank(solver.comm) == 0
        println("density profile: rho(r=", first(r), ") = ", first(rho),
                ", rho(r=", last(r), ") = ", last(rho))
        println(solver.t >= opt.tfinal * (1 - 1e-9) ?
                "reached tfinal = $(opt.tfinal)" :
                "stopped at t = $(solver.t) after nmax = $(opt.nmax)")
    end
    isempty(opt.output) || save_vtk(solver, Q, opt.output)
end
