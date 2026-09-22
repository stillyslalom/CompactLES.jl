# Helium-driven, helium/CO2 shock tube. It uses Euler fluxes plus artificial
# regularization; the numerical transition widths are three local mesh spacings.
#
# Run: julia --project=. examples/shock_tube.jl nx=128 ny=16 tfinal=2e-4 nmax=100
#      mpiexec -n 2 julia --project=. -t 1 examples/shock_tube.jl nx=384 ny=32
# Use boundary=outflow for an NSCBC downstream boundary, or output=shock_final
# to write standard VTK output.

using CompactLES
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

const opt = CompactLES.script_args(ARGS, (nx=768, ny=48, tfinal=2.5e-3, nmax=1_000_000,
                               every=50, boundary="closed", output="");
                        positional=(:nx, :tfinal))
opt.boundary in ("closed", "outflow") ||
    throw(ArgumentError("boundary wants closed or outflow"))

const Lx = 4.0
const Lyz = 0.2
const p_driver = 10 * 101325.0
const p_driven = 101325.0
const temperature0 = 300.0

mpi_main() do
    eos = Nasa9Mixture(read_nasa9(["He", "CO2"]))
    xbc = opt.boundary == "closed" ?
          (SlipWallBC(), SlipWallBC()) :
          (SlipWallBC(), NSCBCOutflowBC(pinf=p_driven))
    problem = Problem(
        name="He-driven RM shock tube",
        eos=eos,
        transport=Transport(mu0=0.0),
        domain=((0.0, Lx), (0.0, Lyz), (0.0, Lyz)),
        bcs=(xbc, PeriodicBC(), PeriodicBC()),
        ic=(x, y, z, h) -> begin
            x_interface = 3.0 + 0.10Lyz * cos(2pi * y / Lyz)
            theta = tanh_blend(x, x_interface, 3h)
            pressure = p_driven + (p_driver - p_driven) *
                       (1 - tanh_blend(x, 2.0, 3h))
            Prim(Y=(1 - theta, theta), p=pressure, T_ion=temperature0)
        end,
    )
    numerics = Numerics(n_global=(opt.nx, opt.ny, 1), art=ArtParams(enabled=true),
                        cfl=0.5, control=StepControl(retries=4, validity=:permissive),
                        filter_interval=1,
                        dims=(MPI.Comm_size(MPI.COMM_WORLD), 1, 1))
    solver, Q = setup(problem, numerics)
    run!(solver, Q; tfinal=opt.tfinal, nmax=opt.nmax,
         callback=ProgressLog(every=opt.every, tfinal=opt.tfinal))

    x, rho = line_profile(solver, Q, :rho)
    if MPI.Comm_rank(solver.comm) == 0
        println("mean density profile: rho(x=", first(x), ") = ", first(rho),
                ", rho(x=", last(x), ") = ", last(rho))
        println(solver.t >= opt.tfinal * (1 - 1e-9) ?
                "reached tfinal = $(opt.tfinal)" :
                "stopped at t = $(solver.t) after nmax = $(opt.nmax)")
    end
    isempty(opt.output) || save_vtk(solver, Q, opt.output)
end
