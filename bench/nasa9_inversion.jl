# Cost per point of the NASA-9 temperature inversion, the Newton solve that
# `recover_primitives!` runs at every point and every Runge–Kutta stage under a
# `Nasa9Mixture` and that makes that model several times the step cost of
# `IdealMixture`.
#
# Two quantities, both in nanoseconds per point:
#
# - `mixture_temperature`: the inversion alone, over a fixed set of states
#   (temperature uniform over [T_min, T_max], mass fractions from a
#   deterministic pseudo-random sequence), with each state's internal energy
#   formed from the same mixture, so the solve converges from the state-based
#   seed exactly as it does inside the solver.
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
# Usage: positional point count, then `key=value` options:
#
#   julia --project=. -t 1 bench/nasa9_inversion.jl
#   julia --project=. -t 1 bench/nasa9_inversion.jl 200000 species=He,CO2
#   julia --project=. -t 1 bench/nasa9_inversion.jl N=48 extrapolate=linear

using CompactLES
using Printf

const CL = CompactLES

const DEFAULTS = (n_points = 100_000, N = 32, repeats = 20,
                  species = "N2,O2,CO2,H2O", T_min = 300.0, T_max = 3000.0,
                  extrapolate = :polynomial)

# A deterministic sequence in [0, 1), so the states do not depend on a Random
# implementation that is not a dependency of this package.
_unit(i, k) = mod(0.6180339887498949 * i + 0.4142135623730951 * k * k, 1.0)

function _composition(n_species, i)
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
    return best / n * 1e9
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

function main(opt)
    names = String.(split(opt.species, ','))
    eos = Nasa9Mixture(names; extrapolate=opt.extrapolate)
    @printf("# species %s, T %.0f-%.0f K, extrapolate=%s, %d threads\n",
            join(names, ","), opt.T_min, opt.T_max, opt.extrapolate,
            Threads.nthreads())
    @printf("mixture_temperature  %8.1f ns/point  (%d points, min of %d)\n",
            time_inversion(eos, opt), opt.n_points, opt.repeats)
    @printf("recover_primitives!  %8.1f ns/point  (%d^3 block, min of %d)\n",
            time_recovery(eos, opt), opt.N, opt.repeats)
    return nothing
end

main(CL.script_args(ARGS, DEFAULTS; positional = (:n_points,)))
