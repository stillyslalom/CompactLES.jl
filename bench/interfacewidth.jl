# The width of the shocked two-gas interface of `shock_interface` (test/cases.jl)
# on several definitions, so that it can be set beside widths reported in
# volume-fraction terms. `width_cells` of the case counts the points with
# 0.05 < Y_air < 0.95 anywhere on the line; at a density ratio R the light gas's
# mass fraction amplifies a volume-fraction deficit by up to R, so a tail or a
# ringing of the heavy gas at the 5e-4 level in V on the air side enters that
# count.
#
#   julia --project=. -t 1 bench/interfacewidth.jl                  # both parts
#   julia --project=. -t 1 bench/interfacewidth.jl history
#   julia --project=. -t 1 bench/interfacewidth.jl configs ratios=5.04,100,1000
#
# Parts: `history` runs the δ⁴ default and the species-only `:d8` split
# (bench/detector_split.jl) at `C_D = 1` to each time of `times` at
# `ratio`, before the shock reaches the interface (t ≈ 0.113) and after;
# `configs` runs every configuration of `CONFIGS` to `SI_T` at each density
# ratio of `ratios`. The columns, from the final profile:
#
#   count      `width_cells`, points with 0.05 < Y_air < 0.95 anywhere
#   core       the same count restricted to the contiguous run through the
#              interface (the point where Y_air is nearest 1/2)
#   Y 5-95     distance between the 0.05 and 0.95 crossings of Y_air nearest
#              the interface, linearly interpolated, in cells
#   V 5-95     the same on the air volume fraction, V = Y/(Y + (1 − Y)/R),
#              which is the mole fraction of the two ideal gases
#   V 10-90    the same between 0.1 and 0.9
#   V grad     max-gradient thickness 1/max|ΔV| in cells
#   TV−1       total variation of Y_air less one, the ringing measure of the
#              appendix tables
#   min Y      most negative mass fraction over the run
#
# For a tanh profile in V, Y is a tanh of the same width displaced by
# w·atanh(At) (Brill, Olson & Bokman 2025, appendix A), so `Y 5-95` equals
# `V 5-95` exactly and the two differ only where the profile is not a tanh.
#
# Settings (`key=value`): ratio (the `history` density ratio), times (comma
# list), ratios (comma list for `configs`), N, nmax, profile (1 prints the
# final Y and V around the interface for each `configs` row), only (comma list
# of `CONFIGS` row numbers that `configs` runs; empty runs every row).
#
# Scratch tooling, like everything else in bench/: it prints tables, asserts
# nothing, and is not part of the gate. The results are written up in
# reference/CALIBRATION_APPENDIX.md, "False activation on smooth fields".

using MPI
MPI.Initialized() || MPI.Init()
using CompactLES
using Printf
const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "cases.jl"))
include(joinpath(@__DIR__, "detector_split.jl"))

const OPTS = CompactLES.script_args(filter(a -> occursin('=', a), ARGS),
                                    (ratio = 100.0, times = "0.1,0.15,0.2,0.25,0.35",
                                     ratios = "5.04,100,1000", N = SI_N,
                                     nmax = 30_000, profile = 0, only = ""))
# D_b = c · G[sensor] takes the local mixture sound speed, which at equal p and
# T is sqrt(R) times larger in the light gas than in the heavy one. The rows
# with a uniform sound speed replace c by a constant in D_b alone (bench only;
# the timestep still reads the local coefficient), to separate that asymmetry
# from the sensor: the pre-shock air value and the pre-shock heavy-gas value
# at the case's density ratio of 100.
const UNIFORM_C = Ref(0.0)
const C_AIR = sqrt(1.4)
const C_HEAVY = sqrt(SI_GAMMA_HEAVY / 100)
@inline function CompactLES._species_diffusivity_point!(D_sp, c, sensor_sp,
                                                        o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        cc = UNIFORM_C[] > 0 ? oftype(c[I], UNIFORM_C[]) : c[I]
        D_sp[I] = cc * max(sensor_sp[I], zero(sensor_sp[I]))
    end
    return nothing
end

const NAMES = filter(a -> !occursin('=', a), ARGS)
const PARTS = isempty(NAMES) ? ["history", "configs"] : NAMES
want(name) = name in PARTS

# (label, species-only split, ArtificialProperties keywords, uniform sound
# speed in D_b; 0 keeps the local one)
const CONFIGS = [("δ4 0.1", false, (detector=:delta4, C_D=0.1), 0.0),
                 ("δ4 1", false, (detector=:delta4, C_D=1.0), 0.0),
                 ("species 0.05", true, (detector=:d8, C_D=0.05), 0.0),
                 ("species 0.3", true, (detector=:d8, C_D=0.3), 0.0),
                 ("species 1", true, (detector=:d8, C_D=1.0), 0.0),
                 ("species 3", true, (detector=:d8, C_D=3.0), 0.0),
                 ("d8 1", false, (detector=:d8, C_D=1.0), 0.0),
                 ("bulk δ4 0.1", false, (species_flux=:bulk, detector=:delta4, C_D=0.1), 0.0),
                 ("species 1 c_air", true, (detector=:d8, C_D=1.0), C_AIR),
                 ("species 1 c_hvy", true, (detector=:d8, C_D=1.0), C_HEAVY),
                 ("species 10 c_hvy", true, (detector=:d8, C_D=10.0), C_HEAVY)]
const HISTORY = CONFIGS[[1, 5]]

# Interpolated position where f crosses `level`, searching outward from index
# `c` in direction `step` (±1); NaN if it never does.
function crossing(x, f, c, level, step)
    i = c
    while 1 <= i + step <= length(f)
        a, b = f[i], f[i+step]
        if (a - level) * (b - level) <= 0 && a != b
            return x[i] + (level - a) / (b - a) * (x[i+step] - x[i])
        end
        i += step
    end
    return NaN
end

# Width between the crossings of `lo` and `hi`, each sought from the centre
# towards the side where f reaches it (f increases or decreases through c).
function band_width(x, f, c, lo, hi)
    up = f[end] > f[1]
    xlo = crossing(x, f, c, lo, up ? -1 : 1)
    xhi = crossing(x, f, c, hi, up ? 1 : -1)
    return abs(xhi - xlo)
end

function core_count(Y, c)
    inside(y) = 0.05 < y < 0.95
    inside(Y[c]) || return 0
    lo, hi = c, c
    while lo > 1 && inside(Y[lo-1]); lo -= 1; end
    while hi < length(Y) && inside(Y[hi+1]); hi += 1; end
    return hi - lo + 1
end

function measures(x, Y, R)
    h = x[2] - x[1]
    V = @. Y / (Y + (1 - Y) / R)
    c = argmin(abs.(Y .- 0.5))
    cv = argmin(abs.(V .- 0.5))
    return (count=count(y -> 0.05 < y < 0.95, Y), core=core_count(Y, c),
            y595=band_width(x, Y, c, 0.05, 0.95) / h,
            v595=band_width(x, V, cv, 0.05, 0.95) / h,
            v1090=band_width(x, V, cv, 0.1, 0.9) / h,
            vgrad=1 / maximum(abs, diff(V)), tv=sum(abs, diff(Y)) - 1, V=V, c=c)
end

function run_si(species_only, kw, R, tfin, cu=0.0)
    SPECIES_ONLY[] = species_only
    UNIFORM_C[] = cu
    try
        r = shock_interface(; art=ArtificialProperties(; kw...), rho_heavy=R,
                            tfin=tfin, N=OPTS.N, nmax=OPTS.nmax)
        return r
    catch err
        # A failed ratio-1000 run raises out of the sound speed before the
        # step control sees it.
        err isa SolverFailure || err isa DomainError || rethrow()
        return nothing
    finally
        SPECIES_ONLY[] = false
        UNIFORM_C[] = 0.0
    end
end

const ROW = Printf.Format("%-17s %7.3g %6.3f | %5d %4d | %6.2f %6.2f %6.2f %6.2f |" *
                          " %7.4f %+8.4f %5d\n")
const HEADER = "config            ratio   t      | count core |" *
               " Y 5-95 V 5-95 V10-90 V grad |   TV−1    min Y steps"

function row(label, species_only, kw, R, tfin, cu=0.0)
    r = run_si(species_only, kw, R, tfin, cu)
    if r === nothing || !r.completed
        @printf("%-13s %7.3g %6.3f | %s\n", label, R, tfin,
                r === nothing ? "lost positivity" : "incomplete at $(r.steps) steps")
        return nothing
    end
    m = measures(r.x, r.Y_air, R)
    Printf.format(stdout, ROW, label, R, tfin, m.count, m.core, m.y595, m.v595, m.v1090,
                  m.vgrad, m.tv, r.worst_min_Y, r.steps)
    return (r, m)
end

function initial_row(R)
    # The initial tanh of `shock_interface`, 2h wide in Y.
    h = 1.0 / (OPTS.N - 1)
    x = collect(range(0.0, 1.0; length=OPTS.N))
    Y = @. 1 - CL.tanh_blend(x, SI_X_IFACE, 2h)
    m = measures(x, Y, R)
    Printf.format(stdout, ROW, "initial", R, 0.0, m.count, m.core, m.y595, m.v595,
                  m.v1090, m.vgrad, m.tv, minimum(Y), 0)
end

function history()
    times = parse.(Float64, split(OPTS.times, ','))
    println("\n=== width against time, density ratio $(OPTS.ratio) ===")
    println(HEADER)
    initial_row(OPTS.ratio)
    for (label, species_only, kw, cu) in HISTORY, t in times
        row(label, species_only, kw, OPTS.ratio, t, cu)
    end
end

function print_profile(r, m)
    lo = max(1, m.c - 8)
    hi = min(length(r.Y_air), m.c + 8)
    for i in lo:hi
        @printf("    i %4d  Y_air %.6f  V_air %.6f\n", i - m.c, r.Y_air[i], m.V[i])
    end
end

function configs()
    ratios = parse.(Float64, split(OPTS.ratios, ','))
    println("\n=== configurations at t = $(SI_T) ===")
    println(HEADER)
    for R in ratios
        initial_row(R)
        rows = isempty(OPTS.only) ? CONFIGS : CONFIGS[parse.(Int, split(OPTS.only, ','))]
        for (label, species_only, kw, cu) in rows
            out = row(label, species_only, kw, R, SI_T, cu)
            OPTS.profile == 1 && out !== nothing && print_profile(out...)
        end
    end
end

function main()
    want("history") && history()
    want("configs") && configs()
    println("\ninterfacewidth complete")
end

main()
