# Taylor–Green vortex at Re = 1600 on a 64³ periodic domain, written against
# the Problem/Numerics frontend.
#
# Run:  julia --project=. -t auto examples/taylor_green.jl
#       mpiexec -n 8 julia --project=. -t 2 examples/taylor_green.jl

using MPI
MPI.Init(threadlevel=:funneled)

using CompactLES

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

num = Numerics(n_global=(64, 64, 64), art=ArtParams(enabled=false),
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

const TFINAL = 1.0
run!(solver, Q; tfinal=TFINAL, nmax=200,
     callback=ProgressLog(every=10, tfinal=TFINAL, label="KE",
                          quantity=kinetic_energy))

MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("done.")
