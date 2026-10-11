# The incompressible solver (src/incompressible.jl) on the two-dimensional
# Taylor–Green vortex, an exact solution whose convective term is a pure
# gradient: the velocity decays as exp(−2νt) and the kinematic pressure is
# (cos 2x + cos 2y) exp(−4νt)/4. Periodic on [0, 2π)², and between slip walls
# on [0, π]², where the walls are symmetry planes of the solution and the run
# reproduces the periodic one at twice the points. The decomposed run against
# the undivided one is the "incompressible" phase of mpi_tests.jl.
#
# Serial. Included by runtests.jl, and runnable on its own:
#
#   julia --project=. test/incompressible_tests.jl

if !@isdefined(CL)
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Test
    const CL = CompactLES
end

function incompressible_tgv(N; walls=false, nu=0.1, t_end=0.5, cfl=0.2)
    L = walls ? π : 2π
    n = walls ? N + 1 : N
    s = CL.IncompressibleSolver((n, n, 1); lengths=(L, L, 1.0),
                                periodic=(!walls, !walls, true), nu=nu,
                                comm=MPI.COMM_SELF)
    CL.set_velocity!(s, (x, y, z) -> (sin(x) * cos(y), -cos(x) * sin(y), 0.0))
    CL.run!(s; t_end=t_end, cfl=cfl)
    F = exp(-2nu * s.t)
    err = 0.0
    for i in (1, 2), I in CL.interior(s.decomp)
        x = CL._position(s, I, i)
        x[i] > L - s.h[i] / 4 && continue          # the slot above a closed end
        exact = i == 1 ? sin(x[1]) * cos(x[2]) * F : -cos(x[1]) * sin(x[2]) * F
        err = max(err, abs(s.U[i][I] - exact))
    end
    # The pressure at t from the projection of the right-hand side there.
    CL._momentum!(s, s.t)
    CL._project!(s, s.R)
    perr = 0.0
    for I in CL.interior(s.decomp)
        x, y, _ = CL._position(s, I, 0)
        perr = max(perr, abs(s.phi[I] - (cos(2x) + cos(2y)) * F^2 / 4))
    end
    return (; s, err, perr)
end

@testset "incompressible solver: Taylor–Green" begin
    coarse = incompressible_tgv(16)
    fine = incompressible_tgv(32)
    order = log2(coarse.err / fine.err)
    porder = log2(coarse.perr / fine.perr)
    @test 5.8 < order < 6.3
    @test 5.8 < porder < 6.3
    @test CL.max_divergence(fine.s) < 1e-12
    # The convective term does no work on this flow, so over a short step
    # the energy falls at the discrete viscous rate, to the relative change
    # 2ν dt of that rate over the step.
    s = fine.s
    K0 = CL.kinetic_energy(s)
    rate = CL.viscous_dissipation(s)
    dt = 1e-5
    CL.step!(s, dt)
    @test abs((K0 - CL.kinetic_energy(s)) / dt - rate) < 1e-5 * rate

    # Slip walls on [0, π]² are symmetry planes of the vortex: N intervals
    # there reproduce 2N periodic points.
    box = incompressible_tgv(16; walls=true)
    @test abs(box.err - fine.err) < 1e-3 * fine.err
    @test CL.max_divergence(box.s) < 1e-12

    # A no-slip box is not a Taylor–Green solution; the projection still
    # holds the divergence at the solve tolerance.
    s = CL.IncompressibleSolver((17, 17, 1); lengths=(π, π, 1.0),
                                periodic=(false, false, true), nu=0.1,
                                walls=:noslip, comm=MPI.COMM_SELF)
    CL.set_velocity!(s, (x, y, z) -> (sin(x) * cos(y), -cos(x) * sin(y), 0.0))
    CL.run!(s; t_end=0.2, cfl=0.2)
    @test CL.max_divergence(s) < 1e-12
    @test CL.kinetic_energy(s) < π^2 / 4

    # The skew-symmetric convective term does no work on the discrete kinetic
    # energy, periodic and between slip walls, where the divergence form does.
    u(x, y, z) = (sin(x) * cos(y) + 0.3 * sin(2y + 0.4) * cos(x),
                  -cos(x) * sin(y) + 0.2 * cos(3x - 1), 0.0)
    for (periodic, n, L) in ((true, 32, 2π), (false, 33, π)), form in (:skew, :divergence)
        s = CL.IncompressibleSolver((n, n, 1); lengths=(L, L, 1.0),
                                    periodic=(periodic, periodic, true),
                                    convection=form, comm=MPI.COMM_SELF)
        CL.set_velocity!(s, u)
        CL.run!(s; t_end=10.0, cfl=0.5, nmax=10)
        CL._momentum!(s, s.t)
        CL._project!(s, s.R)
        dot(f, g) = sum(CL._component_dot(s, i, f[i], g[i]) for i in s.dims)
        work = abs(dot(s.U, s.R)) / sqrt(dot(s.U, s.U) * dot(s.R, s.R))
        form === :skew ? (@test work < 1e-12) : (@test work > 1e-8)
    end

    @test_throws ArgumentError CL.IncompressibleSolver((16, 1, 1); comm=MPI.COMM_SELF)
    @test_throws ArgumentError CL.IncompressibleSolver((16, 16, 1); walls=:open,
                                                       comm=MPI.COMM_SELF)
    @test_throws ArgumentError CL.IncompressibleSolver((16, 16, 1); convection=:rotational,
                                                       comm=MPI.COMM_SELF)
end
