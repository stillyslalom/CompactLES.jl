# Time-dependent boundary forcing demo: a sinusoidally oscillating driver
# (full-state Dirichlet inflow) launching compression waves into quiescent
# gas, non-reflecting NSCBC outflow downstream. Exercises the (x, y, z, t)
# boundary prescription evaluated at RK stage times.
#
# Run:  mpiexec -n 2 julia --project=. -t 2 examples/piston_driver.jl

using MPI
MPI.Init(threadlevel=:funneled)

using CompactLES
using Printf

const p0, ρ0 = 1.0, 1.0
const γ = 1.4
const c0 = sqrt(γ * p0 / ρ0)
const A, f = 0.2 * c0, 2.0            # piston amplitude and frequency

np = MPI.Comm_size(MPI.COMM_WORLD)

driver(x, y, z, t) = Prim(u=(A * sin(2π * f * t), 0.0, 0.0), p=p0, rho=ρ0)

prob = Problem(
    name = "oscillating driver",
    eos = single_species(gamma=γ),
    domain = ((0.0, 2.0), (0.0, 0.1), (0.0, 0.1)),
    bcs = ((DirichletBC(driver), NSCBCOutflowBC(pinf=p0)),
           (PeriodicBC(), PeriodicBC()),
           (PeriodicBC(), PeriodicBC())),
    ic = (x, y, z) -> Prim(u=(0.0, 0.0, 0.0), p=p0, rho=ρ0))

num = Numerics(n_global=(256, 12, 12), art=ArtParams(enabled=true),
               cfl=0.5, dims=(np, 1, 1))

s, Q = setup(prob, num)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

run!(s, Q; tfinal=2.0, nmax=100_000,
     callback=(s, Q) -> (s.step % 50 == 0 && rank == 0 &&
                         @printf("step %5d  t = %6.3f\n", s.step, s.t)))
rank == 0 && println("done.")
