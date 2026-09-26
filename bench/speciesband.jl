# The mass-fraction excursion a converged species interface carries, as a
# function of resolution: the measurement behind `StepControl.species_band`,
# the band beyond which the state validation counts a negative mass fraction.
#
#   julia --project=. bench/speciesband.jl                 # every sweep (~3 min)
#   julia --project=. bench/speciesband.jl advection shock  # named sweeps only
#
# Sweeps: `advection` (uniform advection of a slab whose edges are two cells
# wide at every N, and of one whose edges have a fixed physical width, so the
# first holds the width in cells and the second converges), `shock` (the Mach 1.5 air/SF6
# interface of `shock_interface`, with the mass-fraction bound off as the
# failed control), `slab` (the density-ratio-100 slab of `brill_slab`).
#
# Each row prints the worst excursion over every completed step,
# max over points and species of max(-Y_k, Y_k - 1), and the same at the end,
# with the number of end-state points beyond three candidate bands. Scratch
# tooling: it prints a table and asserts nothing; the conclusions are in
# reference/CALIBRATION_APPENDIX.md.

using CompactLES, Printf, MPI
using CompactLES: padded_index
const CL = CompactLES
MPI.Initialized() || MPI.Init()
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const BANDS = (1e-4, 1e-3, 1e-2, 5e-2)

excursion(y) = max(-y, y - 1, 0.0)

function header(title)
    println("\n", title)
    @printf("%-28s %6s %11s %11s   %s\n", "case", "N", "worst run", "worst end",
            "end points beyond " * join(string.(BANDS), " / "))
end

function row(label, N, worst, Yend)
    ends = excursion.(Yend)
    counts = [count(>(b), ends) for b in BANDS]
    @printf("%-28s %6d %11.3e %11.3e   %s\n", label, N, worst, maximum(ends),
            join(counts, " / "))
end

function advection_sweep()
    header("uniform advection, t = $(MIX_T), one period is t = 1")
    for (label, width) in (("2 cells", N -> 2.0), ("1/32 physical", N -> N / 32))
        for N in (64, 128, 256, 512, 1024)
            worst = Ref(0.0)
            cb = (s, Q) -> begin
                for i in 1:s.decomp.n_local[1]
                    I = padded_index(s, i, 1, 1)
                    worst[] = max(worst[], excursion(Q[I, 1] / (Q[I, 1] + Q[I, 2])))
                end
            end
            _, Y1, _, _, ok = species_advection(; N, delta=width(N), callback=cb)
            ok || println("  incomplete")
            row("advection, " * label, N, worst[], Y1)
        end
    end
end

function shock_sweep()
    header("Mach 1.5 air/SF6, t = $(SI_T)")
    for N in (100, 200, 400, 800, 1600)
        r = shock_interface(; N)
        r.completed || println("  incomplete")
        row("shock, C_Y = 100", N, max(-r.worst_min_Y, r.worst_max_Y - 1), r.Y_air)
    end
    for N in (200, 400, 800)
        r = shock_interface(; N, art=ArtificialProperties(enabled=true, C_Y=0.0))
        row("shock, C_Y = 0", N, max(-r.worst_min_Y, r.worst_max_Y - 1), r.Y_air)
    end
end

function slab_sweep()
    header("slab at density ratio $(BR_R), $(BR_PERIODS) periods")
    # Below Np = 7 the slab is under-resolved and the run loses positivity.
    for Np in (7, 14, 28)
        r = brill_slab(; Np)
        r.completed || println("  incomplete")
        row("slab, Np = $Np", 20Np, max(-r.worst_min_Y, r.worst_max_Y - 1),
            r.Y_light)
    end
end

function main()
    sweeps = Dict("advection" => advection_sweep, "shock" => shock_sweep,
                  "slab" => slab_sweep)
    names = isempty(ARGS) ? ["advection", "shock", "slab"] : ARGS
    for name in names
        haskey(sweeps, name) || error("unknown sweep $name; have $(keys(sweeps))")
        sweeps[name]()
    end
end

main()
