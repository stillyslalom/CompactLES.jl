# The reflection of a spherical wave at a characteristic outflow face: the
# dipole radiated by the sphere of the Oscillating sphere tutorial
# (docs/literate/oscillating_sphere.jl), leaving through an NSCBCOutflowBC at
# the outer radius of a spherical grid resolved in r and the polar angle.
#
#   julia --project=. -t 1 bench/sphereoutflow.jl
#   julia --project=. -t 1 bench/sphereoutflow.jl radii=2.1,4.1 sigmas=0,0.25,1
#   julia --project=. -t 1 bench/sphereoutflow.jl radii=2.1 sigmas=0 ns=64,128,256
#   julia --project=. -t 1 bench/sphereoutflow.jl radii=2.1,4.1 sigmas=0.25 betas=1
#
# The configuration and the measurement are `sphere_dipole_reflection` in
# test/sphere_dipole.jl, which test/serial_suite.jl asserts on one grid: the
# run ends before the reflected wave returns from the sphere, the pressure
# amplitude over the last period on the line of nodes nearest the axis is
# fitted to an outgoing and an incoming spherical Hankel function, and the
# ratio of their coefficients is the face's reflection coefficient. The
# table lists it beside 1/(2kR), the reflection of a face that treats the
# wave as plane, and 1/(2(kR)²), the reflection of the first-order
# radiation condition, both for the exact field, with the fitted outgoing
# amplitude over the exact one.
#
# Settings (`key=value`): radii, sigmas, betas and ns (comma-separated sweeps
# of the outer radius, of NSCBCOutflowBC's `sigma` and `beta_t`, and of the
# radial node count over [0.1, 2.1], which every radius keeps the spacing
# of), ntheta (polar nodes), and samples (per period). Scratch tooling, like
# everything else in bench/: it prints a table and asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using LinearAlgebra
using Printf

const CL = CompactLES

const OPTS = CL.script_args(ARGS, (radii="2.1,4.1", sigmas="0,0.25", betas="-1",
                                   ns="128", ntheta=16, samples=32))
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("sphereoutflow.jl runs serially")

include(joinpath(@__DIR__, "..", "test", "sphere_dipole.jl"))

parselist(T, s) = parse.(T, split(s, ','))

const ROW = Printf.Format("%-6.2f %-7.2f %-7.3f %-6.2f %-5d %-5d %-7d %-10.3e " *
                          "%-10.3e %-10.3e %-8.4f %.1f s\n")

function main()
    println("R      kR      sigma   beta_t n     nr    steps   |β/α|      1/(2kR)    " *
            "1/(2kR²)   |α/A|    wall")
    for R in parselist(Float64, OPTS.radii), n in parselist(Int, OPTS.ns),
        sigma in parselist(Float64, OPTS.sigmas), beta_t in parselist(Float64, OPTS.betas)
        wall = @elapsed m = sphere_dipole_reflection(R; sigma, beta_t, n,
                                                     ntheta=OPTS.ntheta,
                                                     samples=OPTS.samples)
        Printf.format(stdout, ROW, R, m.kR, sigma, beta_t, n, m.nr, m.steps,
                      m.reflection, 1 / (2m.kR), 1 / (2m.kR^2), m.amplitude, wall)
    end
end

main()
