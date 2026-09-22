# A sinusoidally driven one-dimensional piston with an NSCBC downstream end.
#
# Run: julia --project=. examples/piston_driver.jl nx=128 tfinal=0.2 nmax=100
#      mpiexec -n 2 julia --project=. -t 1 examples/piston_driver.jl nx=256

using CompactLES
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

const opt = CompactLES.script_args(ARGS, (nx=256, tfinal=2.0, nmax=100_000,
                               every=50, output=""); positional=(:nx, :tfinal))
const p0 = 1.0
const rho0 = 1.0
const gamma_gas = 1.4
const c0 = sqrt(gamma_gas * p0 / rho0)
const amplitude = 0.2 * c0
const frequency = 2.0

mpi_main() do
    driver(x, y, z, t) =
        Prim(u=(amplitude * sin(2pi * frequency * t), 0.0, 0.0), p=p0, rho=rho0)
    problem = Problem(
        name="oscillating piston",
        eos=IdealSpecies("gas"; R=1.0, gamma=gamma_gas),
        domain=((0.0, 2.0), (0.0, 1.0), (0.0, 1.0)),
        bcs=((DirichletBC(driver), NSCBCOutflowBC(pinf=p0)), PeriodicBC(), PeriodicBC()),
        ic=(x, y, z) -> Prim(u=(0.0, 0.0, 0.0), p=p0, rho=rho0),
    )
    numerics = Numerics(n_global=(opt.nx, 1, 1), art=ArtParams(enabled=true),
                        cfl=0.5, dims=(MPI.Comm_size(MPI.COMM_WORLD), 1, 1))
    solver, Q = setup(problem, numerics)
    run!(solver, Q; tfinal=opt.tfinal, nmax=opt.nmax,
         callback=ProgressLog(every=opt.every, tfinal=opt.tfinal))

    x, pressure = line_profile(solver, Q, :p)
    if MPI.Comm_rank(solver.comm) == 0
        println("pressure profile: p(x=", first(x), ") = ", first(pressure),
                ", p(x=", last(x), ") = ", last(pressure))
        println(solver.t >= opt.tfinal * (1 - 1e-9) ?
                "reached tfinal = $(opt.tfinal)" :
                "stopped at t = $(solver.t) after nmax = $(opt.nmax)")
    end
    isempty(opt.output) || save_vtk(solver, Q, opt.output)
end
