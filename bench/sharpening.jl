# The interface sharpening flux (`ArtificialProperties.C_sharpen`) under the
# production defaults otherwise, on the two instruments it is judged by.
#
#   julia --project=. -t 1 bench/sharpening.jl                         # both parts
#   julia --project=. -t 1 bench/sharpening.jl interface ratios=100,1000
#   julia --project=. -t 1 bench/sharpening.jl slab Ns=64,128,256
#
# `interface` is `shock_interface` of test/cases.jl (a Mach 1.5 shock in air
# into a two-cell interface with a heavy gas) at each density ratio of
# `ratios`, without the flux and at each `C_sharpen` of `gammas` and
# `sharpen_width` of `widths`; `ungated=1` adds the same rows with the gate
# held open, which is the unlocalized form of Brill, Olson & Bokman. Columns,
# from the final profile, as in bench/interfacewidth.jl: `count` the points
# with 0.05 < Y_air < 0.95, `Y 5-95` and `V 5-95` the interpolated widths of
# the air mass and volume fractions in cells, `V 10-90`, `V grad` the
# max-gradient thickness 1/max|ΔV|, TV−1 of Y_air and the worst Y over the
# run. This script does not include bench/detector_split.jl, whose dispatch
# override changes the default detector's arithmetic; its rows are the
# production defaults.
#
# `slab` is the composition slab of bench/falseactivation.jl (molecular-weight
# ratio 5, uniform p, T and u = 1, one period, tanh edges of physical width W)
# over the ladder `Ns`, whose edges are logistic profiles of thickness N·W/2
# cells. Columns: the time maxima of |S|/(ρc) (the light gas's sharpening flux
# along x against the grid scale, read from the `grad_Q` column that holds it
# after the step's last stage) and of the fraction of points where it is
# nonzero; the L1 error of Y_heavy against the exact translation with and
# without the flux; their L1 difference, the flux's deposit; the step counts.
#
# Settings (`key=value`): ratios, gammas, widths, Ns (comma lists), N (the
# interface resolution), W, nmax, ungated (0 or 1).
#
# Scratch tooling, like everything else in bench/: it prints tables and
# asserts nothing. The results are written up in
# reference/CALIBRATION_APPENDIX.md, "The interface sharpening flux".

using MPI
MPI.Initialized() || MPI.Init()
using CompactLES
using CompactLES: padded_index, xcoord
using Printf
const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const OPTS = CompactLES.script_args(filter(a -> occursin('=', a), ARGS),
                                    (ratios = "100,1000", gammas = "0.3,1,3",
                                     widths = "1,1.5", Ns = "64,128,256", N = SI_N,
                                     W = 1 / 16, nmax = 8000, ungated = 0))
const NAMES = filter(a -> !occursin('=', a), ARGS)
const PARTS = isempty(NAMES) ? ["interface", "slab"] : NAMES
list(s) = parse.(Float64, split(s, ','))

# The gate held open for the `ungated` rows.
const UNGATED = Ref(false)
@inline CL._sharpen_gate(θ::T) where {T} =
    UNGATED[] ? one(T) : clamp((θ - T(CL.SHARPEN_GATE[1])) /
                               (T(CL.SHARPEN_GATE[2]) - T(CL.SHARPEN_GATE[1])),
                               zero(T), one(T))

function crossing(x, f, c, level, step)
    i = c
    while 1 <= i + step <= length(f)
        a, b = f[i], f[i+step]
        (a - level) * (b - level) <= 0 && a != b &&
            return x[i] + (level - a) / (b - a) * (x[i+step] - x[i])
        i += step
    end
    return NaN
end

function band(x, f, c, lo, hi)
    up = f[end] > f[1]
    return abs(crossing(x, f, c, hi, up ? 1 : -1) - crossing(x, f, c, lo, up ? -1 : 1))
end

function interface_row(label, R; kw...)
    r = try
        shock_interface(; art=ArtificialProperties(; kw...), rho_heavy=R, nmax=OPTS.nmax)
    catch err
        err isa SolverFailure || err isa DomainError || rethrow()
        nothing
    end
    if r === nothing || !r.completed
        @printf("%-22s %6.4g | %s\n", label, R,
                r === nothing ? "lost positivity" : "incomplete at $(r.steps) steps")
        return
    end
    x, Y = r.x, r.Y_air
    h = x[2] - x[1]
    V = @. Y / (Y + (1 - Y) / R)
    c = argmin(abs.(Y .- 0.5)); cv = argmin(abs.(V .- 0.5))
    @printf("%-22s %6.4g | %5d | %6.2f %6.2f %6.2f %6.2f | %7.4f %+8.4f %5d\n", label, R,
            count(y -> 0.05 < y < 0.95, Y), band(x, Y, c, 0.05, 0.95) / h,
            band(x, V, cv, 0.05, 0.95) / h, band(x, V, cv, 0.1, 0.9) / h,
            1 / maximum(abs, diff(V)), sum(abs, diff(Y)) - 1, r.worst_min_Y, r.steps)
end

function interface()
    println("\n=== the shocked interface at t = $(SI_T), N = $(OPTS.N) ===")
    println("config                  ratio | count | Y 5-95 V 5-95 V10-90 V grad |" *
            "   TV−1    min Y steps")
    for R in list(OPTS.ratios)
        interface_row("unsharpened", R)
        for open in (OPTS.ungated == 1 ? (false, true) : (false,)),
            w in list(OPTS.widths), g in list(OPTS.gammas)
            UNGATED[] = open
            interface_row(@sprintf("C %.2g w %.2g%s", g, w, open ? " ungated" : ""), R;
                          C_sharpen=g, sharpen_width=w)
            UNGATED[] = false
        end
    end
end

slab(x) = (1 - tanh(cos(2π * x) / (2π * OPTS.W)) / tanh(1 / (2π * OPTS.W))) / 2

function slab_run(N, g)
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.4)])
    h = 1.0 / N
    prob = Problem(eos=eos, transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)), bcs=per3,
                   ic=(x, y, z) -> (Yh = slab(x);
                                    Prim(Y=(1 - Yh, Yh), u=(1.0, 0.0, 0.0), p=1.0,
                                         T_ion=1.0)))
    solver, Q = setup(prob, Numerics(n_global=(N, 1, 1),
                                     art=ArtificialProperties(C_sharpen=g)))
    nx = solver.decomp.n_local[1]
    peak = Ref(0.0); frac = Ref(0.0)
    sample = (s, Q) -> begin
        g > 0 || return
        S = s.grad_Q[1, 3]
        n = 0
        for i in 1:nx
            I = padded_index(s, i, 1, 1)
            peak[] = max(peak[], abs(S[I]) / (s.rho[I] * s.c[I]))
            n += S[I] != 0
        end
        frac[] = max(frac[], n / nx)
    end
    run!(solver, Q; tfinal=1.0, nmax=OPTS.nmax, callback=sample)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    xs = Float64[xcoord(solver, 1, i) for i in 1:nx]
    Yh = [solver.Y[2][padded_index(solver, i, 1, 1)] for i in 1:nx]
    return (; x=xs, Y=Yh, peak=peak[], frac=frac[], steps=solver.step,
            ok=completed(solver, 1.0))
end

function slab_part()
    println("\n=== the composition slab, W = $(OPTS.W), one period ===")
    println("   N  N·W/2   C |  |S|/rho c  frac on |  E sharp   E plain  deposit |" *
            " steps sharp plain")
    for N in Int.(list(OPTS.Ns))
        plain = slab_run(N, 0.0)
        for g in list(OPTS.gammas)
            r = slab_run(N, g)
            @printf("%4d %5.1f %5.2g | %10.2e %8.3f | %8.2e %8.2e %8.2e | %5d %5d%s\n",
                    N, N * OPTS.W / 2, g, r.peak, r.frac, l1(r.Y, slab.(r.x)),
                    l1(plain.Y, slab.(plain.x)), l1(r.Y, plain.Y), r.steps,
                    plain.steps, (r.ok && plain.ok) ? "" : "  INCOMPLETE")
        end
    end
end

function main()
    "interface" in PARTS && interface()
    "slab" in PARTS && slab_part()
    println("\nsharpening complete")
end

main()
