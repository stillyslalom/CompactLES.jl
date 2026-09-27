# An acoustic pulse reflected and transmitted at a gas-gas interface, against
# the sharp-interface impedance solution: `interface_pulse` of test/cases.jl,
# which test/validation.jl guards at one resolution, swept here over the grid.
#
#   julia --project=. bench/interfaceacoustics.jl                    # (~2 min)
#   julia --project=. bench/interfaceacoustics.jl n=800 art=on,off delta=4
#
# `cases` selects the light-to-heavy (`lh`) and heavy-to-light (`hl`) pulse,
# `art` the artificial properties, `delta` the interface width in cells,
# `sigma` the pulse width and `amp` its amplitude. Each row prints the
# measured reflection and transmission coefficients and their error against
# (Z_t - Z_i)/(Z_t + Z_i) and 2 Z_t/(Z_t + Z_i), the flux balance
# R^2 + (Z_i/Z_t) T^2, and max |p' - p'_exact| / amp. Scratch tooling: it
# prints a table and asserts nothing; the conclusions are in
# reference/CALIBRATION_APPENDIX.md.

using CompactLES, Printf, MPI
const CL = CompactLES
MPI.Initialized() || MPI.Init()
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const OPT = CompactLES.script_args(ARGS, (n = "400,800,1600,3200", cases = "lh,hl",
    art = "on", amp = 1e-4, sigma = IA_SIGMA, delta = 2.0))

function main()
    ns = parse.(Int, split(OPT.n, ','))
    cases = Symbol.(split(OPT.cases, ','))
    arts = [a == "on" for a in split(OPT.art, ',')]
    @printf("%-4s %-4s %6s %8s %9s %9s %9s %9s %9s %9s %7s\n", "case", "art",
            "N", "cells_t", "R", "R err", "T", "T err", "energy", "err", "steps")
    for case in cases, art in arts, N in ns
        r = interface_pulse(case, N; art, amp=OPT.amp, sigma=OPT.sigma,
                            delta=OPT.delta, nmax=200_000)
        r.completed || println("  incomplete")
        @printf("%-4s %-4s %6d %8.1f %9.5f %9.2e %9.5f %9.2e %9.6f %9.2e %7d\n",
                case, art ? "on" : "off", N, r.cells_t, r.R, r.R - r.Rex, r.T,
                r.T - r.Tex, r.energy, r.err, r.steps)
    end
end

main()
