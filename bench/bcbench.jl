# Cost of the boundary-condition loops, which run over O(N^2) plane points per
# face per RK stage. Sensitive to whether `wallplane` infers concretely.
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

function timeit(f; reps=30)
    f(); f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    t
end

for N in (32, 64)
    # NSCBC outflow + Dirichlet inflow: the heaviest boundary path
    s = Solver(n_global=(N, N, N), L_domain=(1.0, 0.4, 0.4),
               bcs=((DirichletBC((x, y, z, t) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0)),
                     NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
               transport=Transport(mu0=1e-3), art=ArtParams(enabled=true))
    Q = allocate_state(s); dQ = zero(Q)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    trhs = timeit(() -> compute_rhs!(s, Q, dQ))
    tbc  = timeit(() -> apply_bcs!(s, Q))
    @printf("N=%-4d NSCBC+Dirichlet   compute_rhs! %8.3f ms   apply_bcs! %8.4f ms\n",
            N, 1e3trhs, 1e3tbc)

    # slip walls only: same plane loops, much cheaper body
    s2 = Solver(n_global=(N, N, N), L_domain=(1.0, 0.4, 0.4),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                transport=Transport(mu0=1e-3), art=ArtParams(enabled=true))
    Q2 = allocate_state(s2); dQ2 = zero(Q2)
    initialize!(s2, Q2, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    @printf("N=%-4d slip walls        compute_rhs! %8.3f ms   apply_bcs! %8.4f ms\n",
            N, 1e3 * timeit(() -> compute_rhs!(s2, Q2, dQ2)),
            1e3 * timeit(() -> apply_bcs!(s2, Q2)))
end
