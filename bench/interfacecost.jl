# Step cost of the ghost fluxes at patch and level interfaces against the
# closure rows: per case, the same configuration under `interface_flux = :ghost`
# and `:closure`, each advanced in chunks of a few steps through `run!`, the two
# chunks of a round in alternating order, and the median over the rounds of the
# wall time per step. Both solvers live in one process and step their own
# states, so slow drift (thermal, a migration to efficiency cores) affects both
# members of a round alike and the ratio resolves a difference well below the
# 10-20% run-to-run spread of a single timing (CLAUDE.md, Timing noise). The
# per-round ratios are printed beside the medians.
#
# Every case is periodic and smooth, with the artificial properties and the
# filter on unless the case says otherwise, at the default level interpolation
# order of each path (8 under the ghost fluxes, 6 under the closure rows):
#
#   patches        (64, 48, 48) in two patches along x, inviscid
#   patches_mu     the same with `mu = <mu>`
#   box2d          a static 17² box on a 96² root, global step, `mu`
#   tiles2d        a 48² region of a 96² root in sixteen 37-node tiles,
#                  subcycled, `mu`
#   tiles2d_artoff the same with the artificial properties off, where the
#                  coarse-fine ghost fluxes read the shell's gradient ring
#   box3d          a static 13³ box on a 40³ root, global step, `mu`
#
# On a refined case the script also times one shell imposition of every
# transfer (`prolong_level_ghosts!`'s per-transfer step, `BoxFill`), with and
# without the gradient ring, where the transfer carries one: the ring's share of
# the imposition. A global step imposes the shell six times (after each of the
# five stages and after the step), a subcycled level sixteen times per root step
# (five stages of three substeps, and after the step).
#
# Usage, one thread (the cases are small; threads change the ratio):
#
#   julia --project=. -t 1 bench/interfacecost.jl
#   julia --project=. -t 1 bench/interfacecost.jl cases=box2d,tiles2d rounds=15
#
# Options:
#   cases   comma-separated case names (default all, in the order above)
#   rounds  rounds per case (default 9)
#   mu      molecular viscosity of the viscous cases (default 1e-3)
using MPI
MPI.Initialized() || MPI.Init()
using CompactLES
using Printf, Statistics
const CL = CompactLES

const OPTS = CL.script_args(ARGS, (
    cases="patches,patches_mu,box2d,tiles2d,tiles2d_artoff,box3d", rounds=9, mu=1e-3))
const PER = (PeriodicBC(), PeriodicBC())

ic(x, y, z) = Prim(rho=1 + 0.1 * sin(2π * (x + y)) + 0.05 * cos(2π * z),
                   u=(0.3 + 0.1 * sin(2π * y), 0.2 + 0.1 * cos(2π * x), 0.1 * sin(2π * x)),
                   p=1 + 0.05 * cos(2π * x))

# (name, keywords, steps per chunk)
function case_list(mu)
    visc = ConstantTransport(mu0=mu)
    base(n) = (n_global=n, L_domain=(1.0, 1.0, 1.0), bcs=(PER, PER, PER))
    tiles = (; base((96, 96, 1))..., transport=visc, tile=12, subcycle=true,
             refine=BlockRegion((24, 24, 0), (48, 48, 1)))
    return [("patches", (; base((64, 48, 48))..., patch_grid=(2, 1, 1)), 3),
            ("patches_mu", (; base((64, 48, 48))..., patch_grid=(2, 1, 1),
                            transport=visc), 3),
            ("box2d", (; base((96, 96, 1))..., transport=visc,
                       refine=BlockRegion((40, 40, 0), (17, 17, 1))), 10),
            ("tiles2d", tiles, 4),
            ("tiles2d_artoff", (; tiles..., art=ArtificialProperties(enabled=false)), 4),
            ("box3d", (; base((40, 40, 40))..., transport=visc,
                       refine=BlockRegion((13, 13, 13), (13, 13, 13))), 3)]
end

chunk!(s, Q, n) = @elapsed run!(s, Q; tfinal=1e9, nmax=s.step + n)

# One shell imposition of every transfer of every level, with the gradient ring
# (where the transfer carries one) or without it.
function impose_all!(s, Q, ring::Bool)
    levels = getfield(s, :levels)
    patches = getfield(s, :patches)
    for lev in levels[2:end], lt in lev.transfers
        lt.fine_index == 0 && continue
        fine = patches[lt.fine_index]
        CL._impose_shell!(s, Q, lt, CL.BoxFill(), fine.decomp,
                          ring && lt.gradients !== nothing)
    end
    return Q
end

function timecase(name, kw, chunk, rounds)
    solvers = map((:ghost, :closure)) do flux
        s = Solver(; kw..., interface_flux=flux)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        chunk!(s, Q, 2)
        (s, Q)
    end
    tg = Float64[]
    tc = Float64[]
    for r in 1:rounds
        order = isodd(r) ? ((1, tg), (2, tc)) : ((2, tc), (1, tg))
        for (k, acc) in order
            s, Q = solvers[k]
            push!(acc, chunk!(s, Q, chunk) / chunk)
        end
    end
    mg, mc = median(tg), median(tc)
    @printf("%-15s ghost %9.2f ms  closure %9.2f ms  ghost/closure %.3f  (%d-step chunks)\n",
            name, 1e3mg, 1e3mc, mg / mc, chunk)
    @printf("                per-round ghost/closure: %s\n",
            join([@sprintf("%.3f", a / b) for (a, b) in zip(tg, tc)], " "))
    s, Q = solvers[1]
    nlev = length(getfield(s, :levels))
    if nlev > 1
        rings = count(lt -> lt.gradients !== nothing,
                      lt for lev in getfield(s, :levels)[2:end] for lt in lev.transfers)
        with = Float64[]
        without = Float64[]
        for _ in 1:30
            push!(with, @elapsed impose_all!(s, Q, true))
            push!(without, @elapsed impose_all!(s, Q, false))
        end
        ntr = sum(length(lev.transfers) for lev in getfield(s, :levels)[2:end])
        @printf("                one shell imposition: %.3f ms with the gradient ring, ",
                1e3median(with))
        @printf("%.3f ms without (%d of %d transfers carry one)\n",
                1e3median(without), rings, ntr)
    end
    flush(stdout)
end

function main()
    println("=== julia --project=. -t ", Threads.nthreads(), " bench/interfacecost.jl ",
            join(ARGS, " "))
    println("    ", gethostname(), ", ", Sys.CPU_NAME, ", ", Sys.CPU_THREADS,
            " logical CPUs, Julia ", VERSION)
    names = split(OPTS.cases, ',')
    cases = case_list(OPTS.mu)
    unknown = setdiff(names, first.(cases))
    isempty(unknown) || throw(ArgumentError("cases= names no case: " * join(unknown, ", ")))
    for (name, kw, chunk) in cases
        name in names || continue
        timecase(name, kw, chunk, OPTS.rounds)
    end
end
main()
