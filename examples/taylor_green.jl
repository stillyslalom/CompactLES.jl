# Taylor-Green vortex at Re = 1600 on a periodic domain.
#
# Run: julia --project=. examples/taylor_green.jl N=32 tfinal=0.1 nmax=30
#      mpiexec -n 2 julia --project=. -t 1 examples/taylor_green.jl N=64

using CompactLES
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

const opt = CompactLES.script_args(ARGS, (N=64, tfinal=1.0, nmax=200, every=10, output="");
                        positional=(:N, :tfinal))
const reynolds = 1600.0
const gamma_gas = 1.4
const p0 = 10.0^2 / gamma_gas

mpi_main() do
    problem = Problem(
        name="Taylor-Green vortex",
        eos=IdealSpecies("gas"; R=1.0, gamma=gamma_gas),
        transport=Transport(mu0=1.0 / reynolds),
        domain=((0.0, 2pi), (0.0, 2pi), (0.0, 2pi)),
        bcs=(PeriodicBC(), PeriodicBC(), PeriodicBC()),
        ic=(x, y, z) -> Prim(
            u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
            p=p0 + (cos(2x) + cos(2y)) * (cos(2z) + 2) / 16,
            rho=1.0),
    )
    numerics = Numerics(n_global=(opt.N, opt.N, opt.N), art=ArtParams(enabled=false),
                        cfl=0.6, filter_interval=1)
    solver, Q = setup(problem, numerics)
    ke_field = CompactLES.field(solver.decomp)
    function total_kinetic_energy(solver, Q)
        nx, ny, nz = solver.decomp.n_local
        m1, m2, m3 = solver.equations.i_mom
        for k in 1:nz, j in 1:ny, i in 1:nx
            I = gidx(solver, i, j, k)
            ke_field[I] = (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2Q[I, 1])
        end
        volume_integral(solver, ke_field)
    end
    run!(solver, Q; tfinal=opt.tfinal, nmax=opt.nmax,
         callback=ProgressLog(every=opt.every, tfinal=opt.tfinal, label="total KE",
                              quantity=total_kinetic_energy))

    if MPI.Comm_rank(solver.comm) == 0
        println(solver.t >= opt.tfinal * (1 - 1e-9) ?
                "reached tfinal = $(opt.tfinal)" :
                "stopped at t = $(solver.t) after nmax = $(opt.nmax)")
    end
    isempty(opt.output) || save_vtk(solver, Q, opt.output)
end
