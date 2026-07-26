# Taylor–Green vortex at Re = 1600 on a periodic domain, written against the
# Problem/Numerics frontend.
#
# Grid, end time and step cap come from the environment so a batch script or a
# scaling sweep can drive this without editing it — the same reason
# bench/tgv_energy.jl does. A sweep should cap CL_NMAX: the settled step time is
# visible by step 30, and a full run to t = 1 is thousands of steps.
#
# Run:  julia --project=. -t auto examples/taylor_green.jl
#       mpiexec -n 8 julia --project=. -t 2 examples/taylor_green.jl
#       CL_N=256 CL_NMAX=30 srun -n 448 --cpu-bind=threads julia -t 1 ...

using MPI
MPI.Init(threadlevel=:funneled)

using CompactLES

const N      = parse(Int,     get(ENV, "CL_N",      "64"))
const TFINAL = parse(Float64, get(ENV, "CL_TFINAL", "1.0"))
const NMAX   = parse(Int,     get(ENV, "CL_NMAX",   "200"))
const EVERY  = parse(Int,     get(ENV, "CL_EVERY",  "10"))

const Re = 1600.0
const c0 = 10.0                    # sound speed → Mach ≈ 0.1
const γ  = 1.4
const p0 = c0^2 / γ                # ρ0 = 1

prob = Problem(
    name = "taylor-green",
    eos = single_species(gamma=γ),
    transport = Transport(mu0=1.0 / Re),
    domain = ((0.0, 2π), (0.0, 2π), (0.0, 2π)),
    bcs = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
    ic = (x, y, z) -> Prim(
        u = ( sin(x) * cos(y) * cos(z),
             -cos(x) * sin(y) * cos(z),
              0.0),
        p = p0 + (1 / 16) * (cos(2x) + cos(2y)) * (cos(2z) + 2),
        rho = 1.0))

num = Numerics(n_global=(N, N, N), art=ArtParams(enabled=false),
               cfl=0.6, filter_interval=1)

solver, Q = setup(prob, num)

# Preallocated so the diagnostic does not allocate a field per report. Filled
# from Q rather than from solver.rho/u/v/w: the primitives hold the last RK
# stage, not the completed step.
const ke_field = field(solver.decomp)

function kinetic_energy(solver, Q)
    nx, ny, nz = solver.decomp.n_local
    m1, m2, m3 = solver.equations.i_mom
    for k in 1:nz, j in 1:ny, i in 1:nx
        I = gidx(solver, i, j, k)
        ke_field[I] = 0.5 * (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / Q[I, 1]
    end
    # volume_integral carries the reduction and the quadrature weights.
    return volume_integral(solver, ke_field)
end

mpi_main() do
    run!(solver, Q; tfinal=TFINAL, nmax=NMAX,
         callback=ProgressLog(every=EVERY, tfinal=TFINAL, label="KE",
                              quantity=kinetic_energy))
end

MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("done.")
