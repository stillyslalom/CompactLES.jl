# The three-dimensional Taylor–Green vortex on the incompressible solver
# (src/incompressible.jl), with no filter and no model: the kinetic energy,
# its rate of loss −dK/dt, and the part of that loss the discrete viscous
# term accounts for. The remainder, `scheme`, is what the convective term and
# the time integration remove: near zero under the skew-symmetric form,
# which does no work on the kinetic energy, and negative under the
# divergence form where the grid under-resolves the cascade. The peak of
# −dK/dt and its time are printed at the end for comparison with the
# spectral reference at Re = 1600 (Brachet et al., JFM
# 130, 1983; the 512³ data of the high-order workshop), the same case
# `bench/tgv_energy.jl` runs on the compressible solver.
#
# Run from the checkout root:
#   julia --project=. bench/incompressible_tgv.jl [N] [tfinal] [key=value ...]
#   mpiexec -n 8 julia --project=. -t 1 bench/incompressible_tgv.jl 64
#
# Keys:
#   N        grid points per direction on [0, 2π)³ (32)
#   tfinal   end time (10.0)
#   Re       Reynolds number on the unit velocity and length (1600.0)
#   cfl      Courant number of `compute_dt` (0.8)
#   convection  skew (default) or divergence
#   every    print interval in steps (20)
#   nmax     step limit (100000)

using CompactLES, MPI, Printf

const DEFAULTS = (N=32, tfinal=10.0, Re=1600.0, cfl=0.8, convection=:skew, every=20,
                  nmax=100000)

function main(opt)
    C = CompactLES
    N = opt.N
    s = C.IncompressibleSolver((N, N, N); lengths=(2π, 2π, 2π), nu=1 / opt.Re,
                               convection=opt.convection)
    C.set_velocity!(s, (x, y, z) -> (sin(x) * cos(y) * cos(z),
                                     -cos(x) * sin(y) * cos(z), 0.0))
    root = MPI.Comm_rank(MPI.COMM_WORLD) == 0
    volume = (2π)^3
    K = C.kinetic_energy(s) / volume
    t_prev, K_prev = s.t, K
    peak, t_peak = 0.0, 0.0
    root && @printf("%8s %6s %12s %12s %12s %12s %6s\n", "t", "step", "K", "-dK/dt",
                    "viscous", "scheme", "its")
    wall = @elapsed while s.t < opt.tfinal && s.step < opt.nmax
        dt = min(C.compute_dt(s; cfl=opt.cfl), opt.tfinal - s.t)
        C.step!(s, dt)
        K = C.kinetic_energy(s) / volume
        rate = -(K - K_prev) / (s.t - t_prev)
        if rate > peak
            peak, t_peak = rate, (s.t + t_prev) / 2
        end
        if s.step % opt.every == 0 || s.t >= opt.tfinal
            viscous = C.viscous_dissipation(s) / volume
            root && @printf("%8.4f %6d %12.6e %12.6e %12.6e %12.4e %6.1f\n", s.t,
                            s.step, K, rate, viscous, rate - viscous,
                            s.iterations)
            root && flush(stdout)
        end
        t_prev, K_prev = s.t, K
    end
    root && @printf("peak -dK/dt %.5e at t = %.3f; %d steps\n", peak, t_peak, s.step)
    root && @printf("%.1f CG iterations per stage, %.1f s on %d ranks\n",
                    s.total_iterations / (5 * s.step), wall,
                    MPI.Comm_size(MPI.COMM_WORLD))
end

const _opt = CompactLES.script_args(ARGS, DEFAULTS; positional=(:N, :tfinal))
mpi_main(() -> main(_opt))
