# Cylindrically converging shock: an effectively 1-D radial problem
# (n_global = (Nr, 1, 1)) in cylindrical coordinates with the regularized axis.
# A high-pressure outer annulus drives a shock toward r = 0, which reflects
# off the axis and returns — a Guderley-flavored stress test of the axis
# treatment and the collapsed-dimension path. Costs ~Nr points, not Nr×9×9.
#
# Run:  julia --project=. -t auto examples/converging_shock.jl
#       mpiexec -n 4 julia --project=. -t 1 examples/converging_shock.jl

using MPI
MPI.Init(threadlevel=:funneled)

using CompactLES
using Printf

const Nr = 1024
const R  = 1.0
np = MPI.Comm_size(MPI.COMM_WORLD)

prob = Problem(
    name = "converging shock",
    eos = IdealSpecies("gas"; R=1.0, gamma=1.4),
    metric = CylindricalMetric(),
    domain = ((0.0, R), (0.0, 1.0), (0.0, 1.0)),   # θ, z collapsed below
    bcs = ((AxisBC(), SlipWallBC()),
           (PeriodicBC(), PeriodicBC()),
           (PeriodicBC(), PeriodicBC())),
    ic = (r, θ, z) -> begin
        drive = tanh_blend(r, 0.7R, 0.01R)          # 0 inside, 1 outside
        Prim(rho = 1.0 + 3.0 * drive,
             p   = 1.0 + 19.0 * drive)
    end)

num = Numerics(n_global=(Nr, 1, 1), art=ArtParams(enabled=true),
               cfl=0.4, filter_interval=1, dims=(np, 1, 1))

solver, Q = setup(prob, num)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

function diag(solver, Q)
    solver.step % 100 == 0 || return
    nx = solver.decomp.n_local[1]
    pmax = -Inf
    for i in 1:nx
        I = gidx(solver, i, 1, 1)
        ρ = Q[I, 1]
        u = Q[I, 2] / ρ
        p = 0.4 * (Q[I, 5] - 0.5 * ρ * u^2)
        pmax = max(pmax, p)
    end
    pmax = MPI.Allreduce(pmax, max, solver.decomp.comm)
    rank == 0 && @printf("step %6d  t = %8.5f  p_max = %10.4f\n",
                         solver.step, solver.t, pmax)
end

run!(solver, Q; tfinal=0.35, nmax=1_000_000, callback=diag)

open(@sprintf("conv_rank%03d.dat", rank), "w") do io
    for i in 1:solver.decomp.n_local[1]
        I = gidx(solver, i, 1, 1)
        @printf(io, "%.8e  %.8e\n", xcoord(solver, 1, i), Q[I, 1])
    end
end
rank == 0 && println("done; concatenate conv_rank*.dat for ρ(r).")
