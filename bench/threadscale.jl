# How much of the hot-path allocation is Threads.@threads overhead?
# Run at several -t values; per-call bytes should be flat if the allocation is
# real work, and linear in nthreads if it is task/closure churn.
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 48
s = Solver(n_global=(N, N, N), L_domain=(2π, 2π, 2π), bcs=per3,
           transport=Transport(mu0=1e-3), art=ArtParams(enabled=true))
Q = allocate_state(s)
initialize!(s, Q, (x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0))
dQ = zero(Q); du = zero(Q)

probe(f) = (f(); f(); min(@allocated(f()), @allocated(f())))

@printf("threads=%-3d N=%-4d  primitives!=%9d  deriv=%8d  artificial=%9d  rhs=%9d  step=%10d\n",
        Threads.nthreads(), N,
        probe(() -> CL.primitives!(s, Q)),
        probe(() -> CL.deriv_along!(s.tmp_a, s.rho, s, 1, 1)),
        probe(() -> CL.compute_artificial!(s, Q)),
        probe(() -> compute_rhs!(s, Q, dQ)),
        probe(() -> step!(s, Q, dQ, du, 1e-4)))
