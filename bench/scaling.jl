# Wall-time scaling of compute_rhs! with grid size and thread count.
#
#   julia --project=. -t <n> bench/scaling.jl
#
# Threading overhead is a fixed cost per @threads region per thread, so it
# shows up as poor (or negative) scaling at small N and good scaling at large
# N. The crossover tells us where threading starts paying for itself.
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

function timeit(f; reps=20)
    f(); f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    t
end

println("threads = ", Threads.nthreads())
@printf("%-8s %10s %12s %12s\n", "N", "points", "rhs [ms]", "ns/point")
for N in (24, 32, 48, 64, 96)
    s = Solver(nglob=(N, N, N), Ldom=(2π, 2π, 2π), bcs=per3,
               transport=Transport(mu0=1e-3), art=ArtParams(enabled=true))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0))
    dQ = zero(Q)
    t = timeit(() -> compute_rhs!(s, Q, dQ))
    npt = N^3
    @printf("%-8d %10d %12.3f %12.2f\n", N, npt, 1e3t, 1e9t / npt)
end

# 1-D radial (converging_shock.jl shape): threading overhead should dominate
sf = Solver(nglob=(512, 1, 1), Ldom=(1.0, 1.0, 1.0), metric=CylindricalMetric(),
            bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
            art=ArtParams(enabled=true))
Qf = allocate_state(sf)
initialize!(sf, Qf, (r, θ, z) -> Prim(u=(0, 0, 0), p=1 + exp(-40(r - 0.4)^2), rho=1.0))
dQf = zero(Qf)
t = timeit(() -> compute_rhs!(sf, Qf, dQf))
@printf("%-8s %10d %12.3f %12.2f\n", "1D-512", 512, 1e3t, 1e9t / 512)
