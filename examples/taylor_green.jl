# Taylor–Green vortex at Re = 1600 on a 64³ periodic domain, written against
# the Problem/Numerics frontend.
#
# Run:  julia --project=. -t auto examples/taylor_green.jl
#       mpiexec -n 8 julia --project=. -t 2 examples/taylor_green.jl

using MPI
MPI.Init(threadlevel=:funneled)

using CompactLES
using Printf

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

num = Numerics(nglob=(64, 64, 64), art=ArtParams(enabled=false),
               cfl=0.6, filter_interval=1)

s, Q = setup(prob, num)

const cellvol = prod(s.h)
rank0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0

function diag(s, Q)
    s.step % 10 == 0 || return
    ke = 0.0
    nx, ny, nz = s.dec.nloc
    m1, m2, m3 = s.mom
    for k in 1:nz, j in 1:ny, i in 1:nx
        ρ = Q[gidx(s, i, j, k), 1]
        ke += 0.5 * (Q[gidx(s, i, j, k), m1]^2 + Q[gidx(s, i, j, k), m2]^2 +
                     Q[gidx(s, i, j, k), m3]^2) / ρ
    end
    ke = MPI.Allreduce(ke * cellvol, +, s.dec.comm)
    rank0 && @printf("step %5d  t = %8.4f  KE = %.8e\n", s.step, s.t, ke)
end

run!(s, Q; tfinal=1.0, nmax=200, callback=diag)
rank0 && println("done.")
