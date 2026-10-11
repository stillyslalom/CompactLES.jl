# The two-dimensional checks of the incompressible solver
# (src/incompressible.jl).
#
#   part=order   The Taylor–Green vortex, an exact solution whose convective
#                term is a pure gradient, periodic on [0, 2π)² and between
#                slip walls on [0, π]², at N = 16, 32, 64 to t = 1: the
#                velocity and pressure errors and their orders, the
#                conjugate-gradient iterations per stage, and the divergence.
#   part=energy  The work of the projected inviscid right-hand side on the
#                kinetic energy relative to ‖u‖ ‖P N‖, and the energy change
#                over 50 steps at cfl 0.5, under both convective forms, on a
#                periodic plane, a slip box, a no-slip box and a 3-D box with
#                slip walls in z.
#
# Run from the checkout root:
#   julia --project=. -t 1 bench/incompressible_checks.jl part=order
#   julia --project=. -t 1 bench/incompressible_checks.jl part=energy
#
# Keys: part (order), convection (skew; the order part's form), rtol (1e-12).

using CompactLES, MPI, Printf

const DEFAULTS = (part="order", convection=:skew, rtol=1e-12)
const C = CompactLES

function tgv(N; walls, convection, rtol, nu=0.1, t_end=1.0, cfl=0.2)
    L = walls ? π : 2π
    n = walls ? N + 1 : N
    s = C.IncompressibleSolver((n, n, 1); lengths=(L, L, 1.0),
                               periodic=(!walls, !walls, true), nu=nu,
                               convection=convection, rtol=rtol)
    C.set_velocity!(s, (x, y, z) -> (sin(x) * cos(y), -cos(x) * sin(y), 0.0))
    C.run!(s; t_end=t_end, cfl=cfl)
    F = exp(-2nu * s.t)
    err = 0.0
    for i in (1, 2), I in C.interior(s.decomp)
        x = C._position(s, I, i)
        x[i] > L - s.h[i] / 4 && continue          # the slot above a closed end
        exact = i == 1 ? sin(x[1]) * cos(x[2]) * F : -cos(x[1]) * sin(x[2]) * F
        err = max(err, abs(s.U[i][I] - exact))
    end
    C._momentum!(s, s.t)
    C._project!(s, s.R)
    perr = 0.0
    for I in C.interior(s.decomp)
        x, y, _ = C._position(s, I, 0)
        perr = max(perr, abs(s.phi[I] - (cos(2x) + cos(2y)) * F^2 / 4))
    end
    return (; err, perr, its=s.total_iterations / (5s.step), div=C.max_divergence(s))
end

function order(opt)
    for walls in (false, true)
        println(walls ? "slip-wall box [0, π]²" : "periodic [0, 2π)²")
        prev = nothing
        for N in (16, 32, 64)
            r = tgv(N; walls, convection=opt.convection, rtol=opt.rtol)
            ou = prev === nothing ? NaN : log2(prev.err / r.err)
            op = prev === nothing ? NaN : log2(prev.perr / r.perr)
            @printf("  N = %3d  u %.3e (%.2f)  p %.3e (%.2f)  CG/stage %.1f  div %.1e\n",
                    N, r.err, ou, r.perr, op, r.its, r.div)
            prev = r
        end
    end
end

u2(x, y, z) = (sin(x) * cos(y) + 0.3 * sin(2y + 0.4) * cos(x),
               -cos(x) * sin(y) + 0.2 * cos(3x - 1), 0.0)
u3(x, y, z) = (sin(x) * cos(y) * cos(z) + 0.2 * sin(y + 2z), -cos(x) * sin(y) * cos(z),
               0.3 * cos(x - y) * sin(z))

function energy(opt)
    cases = (("periodic 32²", (32, 32, 1), (true, true, true), (2π, 2π, 1.0), u2, :slip),
             ("slip box 33²", (33, 33, 1), (false, false, true), (π, π, 1.0), u2, :slip),
             ("no-slip box 33²", (33, 33, 1), (false, false, true), (π, π, 1.0), u2,
              :noslip),
             ("3-D slip z 16²×17", (16, 16, 17), (true, true, false), (2π, 2π, π), u3,
              :slip))
    for (label, n, periodic, lengths, u, walls) in cases, form in (:skew, :divergence)
        s = C.IncompressibleSolver(n; lengths, periodic, walls, convection=form,
                                   rtol=opt.rtol)
        C.set_velocity!(s, u)
        C._momentum!(s, 0.0)
        C._project!(s, s.R)
        dot(f, g) = sum(C._component_dot(s, i, f[i], g[i]) for i in s.dims)
        work = dot(s.U, s.R) / sqrt(dot(s.U, s.U) * dot(s.R, s.R))
        K0 = C.kinetic_energy(s)
        C.run!(s; t_end=100.0, cfl=0.5, nmax=50)
        @printf("%-20s %-10s  work %+.2e  K/K0 - 1 after 50 steps %+.2e\n", label, form,
                work, C.kinetic_energy(s) / K0 - 1)
    end
end

function main(opt)
    opt.part == "order" ? order(opt) :
    opt.part == "energy" ? energy(opt) :
    throw(ArgumentError("part must be order or energy, not $(opt.part)"))
end

const _opt = CompactLES.script_args(ARGS, DEFAULTS)
mpi_main(() -> main(_opt))
