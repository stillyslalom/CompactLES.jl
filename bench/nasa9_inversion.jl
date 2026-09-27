# The NASA-9 temperature inversion: its cost per point, and where it fails.
# `recover_primitives!` runs the inversion's Newton solve at every point and
# every Runge–Kutta stage under a `Nasa9Mixture`, and that solve makes the
# model several times the step cost of `IdealMixture`.
#
# `part=timing` (default) measures two quantities, both in nanoseconds per
# point:
#
# - `mixture_temperature`: the inversion alone, over a fixed set of states
#   (temperature uniform over [T_min, T_max], mass fractions from a
#   deterministic pseudo-random sequence), with each state's internal energy
#   formed from the same mixture, so the solve converges from the state-based
#   seed exactly as it does inside the solver. The mean Newton iteration count
#   over the same states is printed beside it.
# - `recover_primitives!`: the whole per-point recovery on an `N`³ periodic
#   block initialized with a smooth field over the same temperature range and
#   compositions, which adds the mass fractions, the mixture cp and the stores.
#
# Each is the minimum over `repeats` timed passes after one warm-up pass; the
# minimum is the least noisy estimate of a loop that allocates nothing. The
# temperature range defaults to 300–3000 K so every state crosses the 1000 K
# join of the bundled CEA fits. Run single-threaded: the quantity is a cost per
# point, and `recover_primitives!` would otherwise divide over threads.
#
# A before/after comparison belongs in one session with the same arguments,
# alternating the two checkouts over at least three processes, since the
# run-to-run spread on a desktop is 10–20% (CLAUDE.md, Timing noise).
#
# `part=sweep` counts failures instead. For each species set in `sets`
# (semicolon-separated; a single species is the pure gas, several take the
# pseudo-random compositions above), in Float64 and Float32 and under the
# `:polynomial` and `:linear` extrapolation policies, it inverts `n_points`
# energies formed at temperatures uniform over the span of `bands` and reports,
# per band: the states whose status carries `TEMPERATURE_NOT_CONVERGED` and
# `TEMPERATURE_NO_BRACKET`, the states recovered more than `wrong` (relative)
# from the temperature they were formed at, those of them whose status carries
# no flag at all (a silent failure), the mean and largest iteration count, and
# the largest relative error. The fits are continuous at an interval
# join only to the reader's tolerance, so the error near a join is bounded by
# that continuity and not by the inversion.
#
# Usage: positional point count, then `key=value` options:
#
#   julia --project=. -t 1 bench/nasa9_inversion.jl
#   julia --project=. -t 1 bench/nasa9_inversion.jl 200000 species=He,CO2
#   julia --project=. -t 1 bench/nasa9_inversion.jl N=48 extrapolate=polynomial
#   julia --project=. bench/nasa9_inversion.jl 200000 part=sweep
#   julia --project=. bench/nasa9_inversion.jl part=sweep sets="CO2;N2" bands=6000,20000

using CompactLES
using CompactLES: recover_primitives!
using Printf

const CL = CompactLES

const DEFAULTS = (n_points = 100_000, N = 32, repeats = 20,
                  species = "N2,O2,CO2,H2O", T_min = 300.0, T_max = 3000.0,
                  extrapolate = :linear, part = :timing,
                  sets = "N2;CO2;He;N2,O2,CO2,H2O",
                  bands = "200,1000,6000,13000,20000", wrong = 1e-3)

# A deterministic sequence in [0, 1), so the states do not depend on a Random
# implementation that is not a dependency of this package.
_unit(i, k) = mod(0.6180339887498949 * i + 0.4142135623730951 * k * k, 1.0)

function _composition(n_species, i)
    n_species == 1 && return (1.0,)
    w = ntuple(k -> 0.05 + _unit(i, k), n_species)
    return w ./ sum(w)
end

function time_inversion(eos, opt)
    n_species = nspecies(eos)
    n = opt.n_points
    Y = Matrix{Float64}(undef, n_species, n)
    e = Vector{Float64}(undef, n)
    for i in 1:n
        y = _composition(n_species, i)
        Y[:, i] .= y
        T_ion = opt.T_min + (opt.T_max - opt.T_min) * _unit(i, 0)
        e[i] = sum(y[k] * CL.species_energy(eos, k, T_ion) for k in 1:n_species)
    end
    T_out = similar(e)
    pass() = @inbounds for i in 1:n
        T_out[i] = CL.mixture_temperature(eos, e[i], k -> Y[k, i])
    end
    pass()
    best = Inf
    for _ in 1:opt.repeats
        best = min(best, @elapsed pass())
    end
    iterations = sum(CL._nasa9_temperature(eos, e[i], k -> Y[k, i])[3] for i in 1:n)
    return best / n * 1e9, iterations / n
end

function time_recovery(eos, opt)
    n_species = nspecies(eos)
    N = opt.N
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    solver = Solver(n_global=(N, N, N), L_domain=(1.0, 1.0, 1.0), bcs=per3,
                    eos=eos)
    Q = allocate_state(solver)
    span = opt.T_max - opt.T_min
    initialize!(solver, Q, (x, y, z) -> begin
        s = (sinpi(2x) * sinpi(2y) * sinpi(2z) + 1) / 2
        w = ntuple(k -> 1.05 + sinpi(2 * (x + k * y + k * k * z)), n_species)
        Prim(Y=w ./ sum(w), u=(10.0, 0.0, 0.0), p=1.0e5,
             T_ion=opt.T_min + span * s)
    end)
    CL.exchange_state!(Q, solver.decomp)
    CL.recover_primitives!(solver, eos, Q)
    best = Inf
    for _ in 1:opt.repeats
        best = min(best, @elapsed CL.recover_primitives!(solver, eos, Q))
    end
    return best / length(solver.rho) * 1e9
end

function timing(opt)
    names = String.(split(opt.species, ','))
    eos = Nasa9Mixture(names; extrapolate=opt.extrapolate)
    @printf("# species %s, T %.0f-%.0f K, extrapolate=%s, %d threads\n",
            join(names, ","), opt.T_min, opt.T_max, opt.extrapolate,
            Threads.nthreads())
    ns, iterations = time_inversion(eos, opt)
    @printf("mixture_temperature  %8.1f ns/point  (%d points, min of %d)\n",
            ns, opt.n_points, opt.repeats)
    @printf("Newton iterations    %8.3f per point\n", iterations)
    @printf("recover_primitives!  %8.1f ns/point  (%d^3 block, min of %d)\n",
            time_recovery(eos, opt), opt.N, opt.repeats)
    return nothing
end

# One species set, precision and policy: counts per temperature band.
function sweep_case(::Type{T}, names, policy, edges, opt) where {T}
    eos = Nasa9Mixture(T, names; extrapolate=policy)
    n_species = length(names)
    nb = length(edges) - 1
    count = zeros(Int, nb)
    not_converged = zeros(Int, nb)
    no_bracket = zeros(Int, nb)
    wrong = zeros(Int, nb)
    silent = zeros(Int, nb)
    iterations = zeros(Int, nb)
    max_iterations = zeros(Int, nb)
    max_error = zeros(Float64, nb)
    span = edges[end] - edges[1]
    for i in 1:opt.n_points
        y = T.(_composition(n_species, i))
        T_ref = T(edges[1] + span * _unit(i, 0))
        e = sum(y[k] * CL.species_energy(eos, k, T_ref) for k in 1:n_species)
        T_ion, status, iters = CL._nasa9_temperature(eos, e, k -> y[k])
        b = clamp(searchsortedlast(edges, Float64(T_ref)), 1, nb)
        err = abs(Float64(T_ion) - Float64(T_ref)) / Float64(T_ref)
        count[b] += 1
        not_converged[b] += (status & CL.TEMPERATURE_NOT_CONVERGED) != 0
        no_bracket[b] += (status & CL.TEMPERATURE_NO_BRACKET) != 0
        wrong[b] += !(err <= opt.wrong)
        silent[b] += !(err <= opt.wrong) && status == CL.TEMPERATURE_OK
        iterations[b] += iters
        max_iterations[b] = max(max_iterations[b], iters)
        max_error[b] = max(max_error[b], err)
    end
    for b in 1:nb
        @printf("%-16s %-7s %-10s %5.0f-%5.0f %7d %7d %7d %7d %6d %6.2f %4d %9.2e\n",
                join(names, ","), T, policy, edges[b], edges[b+1], count[b],
                not_converged[b], no_bracket[b], wrong[b], silent[b],
                iterations[b] / max(count[b], 1), max_iterations[b], max_error[b])
    end
    return sum(not_converged), sum(no_bracket), sum(wrong), sum(silent)
end

function sweep(opt)
    edges = parse.(Float64, split(opt.bands, ','))
    @printf("# %d states per case, T %.0f-%.0f K, wrong above %.1e relative\n",
            opt.n_points, edges[1], edges[end], opt.wrong)
    @printf("%-16s %-7s %-10s %11s %7s %7s %7s %7s %6s %6s %4s %9s\n",
            "species", "type", "policy", "band K", "states", "notconv", "nobrack",
            "wrong", "silent", "iters", "max", "max err")
    totals = zeros(Int, 4)
    for set in split(opt.sets, ';'), T in (Float64, Float32),
        policy in (:polynomial, :linear)
        totals .+= sweep_case(T, String.(split(set, ',')), policy, edges, opt)
    end
    @printf("# totals: not converged %d, no bracket %d, wrong %d, silent %d\n",
            totals...)
    return nothing
end

function main(opt)
    opt.part === :timing && return timing(opt)
    opt.part === :sweep && return sweep(opt)
    throw(ArgumentError("part must be timing or sweep, got $(opt.part)"))
end

main(CL.script_args(ARGS, DEFAULTS; positional = (:n_points,)))
