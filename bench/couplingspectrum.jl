# The spectrum of the one-step map of a refined run, linearized about its
# state: whether the level coupling is stable.
#
#   julia --project=. bench/couplingspectrum.jl case=nest N=36
#   julia --project=. bench/couplingspectrum.jl case=level N=72 flux=closure
#
# --- Why this exists ---------------------------------------------------------
#
# A convergence row reads an error at one time, which cannot tell a coupling
# that is merely inaccurate from one that amplifies. This script linearizes
# the map from the state after one step to the state `nmap` steps later by
# central differences in every interior value of every patch (all conserved
# components), and prints the eigenvalues of largest modulus: |λ|, the growth
# rate log|λ|/(nmap dt) per unit time, and the share of the eigenvector on
# each patch. The uniform run at the root spacing is the control, and its
# leading rate comes from the nonlinear base state, not from any coupling.
# The measurements are in reference/CALIBRATION_APPENDIX.md under "Levels
# regridded onto a fold and nested at one".
#
# Cases, from test/smooth_cases.jl, on the standing wave between symmetry
# planes: `uniform` (`wall_case(N; folded = true)`), `level`
# (`plane_level_case`, one level at the plane) and `nest` (`plane_nest_case`,
# two nested levels at the plane). The step is 0.4/(spn N), the one the nest's
# convergence row takes at spn = 18. Each case keeps its own `filter_interval`
# (0) unless `filt` is given; with a cadence above 1 pass `nmap` equal to it,
# since a map over fewer steps than the cadence may contain no filter pass.
#
# Cost: 2n·nmap steps, n being five times the number of interior nodes: about
# 1 s at N = 36 and 5 s at N = 144 for the nest with nmap = 1, plus a dense
# eigensolve of order n.
# Scratch tooling, like everything else in bench/: it prints and asserts
# nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES, Printf, LinearAlgebra
using CompactLES: padded_index, xcoord

const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

const OPTS = CL.script_args(ARGS, (case="nest", N=36, spn=18, flux="ghost", filt=-1,
                                   nmap=1, eps=1e-7, top=6))

function build(o)
    kw = (interface_flux=Symbol(o.flux),)
    o.filt >= 0 && (kw = merge(kw, (filter_interval=o.filt,)))
    o.case == "nest" && return plane_nest_case(o.N; kw...)
    o.case == "level" && return plane_level_case(o.N; kw...)
    o.case == "uniform" &&
        return wall_case(o.N; folded=true, slip=true, cfl=0.9,
                         (o.filt >= 0 ? (filter_interval=o.filt,) : (;))...)
    error("case must be uniform, level or nest, got $(o.case)")
end

function main(o)
    solver, states = build(o)
    SV = states isa AbstractVector ? states : [states]
    patches = getfield(solver, :patches)
    dt = 0.4 / (o.spn * o.N)
    run!(solver, states; tfinal=dt)
    base = [copy(Q) for Q in SV]
    t0 = solver.t
    s0 = solver.step
    saved = (solver.dt_prev, solver.rate_prev, solver.filter_rate_prev)
    dofs = Tuple{Int,CartesianIndex{3},Int}[]
    for (pi, p) in enumerate(patches)
        ps = CL.PatchSolver(solver, p)
        for i in 1:ps.decomp.n_local[1], c in 1:solver.equations.n_cons
            push!(dofs, (pi, padded_index(ps, i, 1, 1), c))
        end
    end
    n = length(dofs)
    # Every call restarts from the same state, clock and step counter, so the
    # map is the same function of `x` each time.
    function advance(x)
        for (k, Q) in enumerate(SV)
            copyto!(Q, base[k])
        end
        for (j, (pi, I, c)) in enumerate(dofs)
            SV[pi][I, c] += x[j]
        end
        solver.t = t0
        solver.step = s0
        solver.dt_prev, solver.rate_prev, solver.filter_rate_prev = saved
        for k in 1:o.nmap
            run!(solver, states; tfinal=t0 + k * dt)
        end
        return [SV[pi][I, c] for (pi, I, c) in dofs]
    end
    J = zeros(n, n)
    x = zeros(n)
    elapsed = @elapsed for j in 1:n
        x[j] = o.eps
        yp = advance(x)
        x[j] = -o.eps
        ym = advance(x)
        x[j] = 0.0
        J[:, j] .= (yp .- ym) ./ (2 * o.eps)
    end
    E = eigen(J)
    order = sortperm(abs.(E.values); rev=true)
    @printf("%s, N = %d, dt = %.3e, flux = %s, filter_interval = %d, %d patches, n = %d\n",
            o.case, o.N, dt, o.flux, solver.filter_interval, length(patches), n)
    @printf("  map linearized in %.1f s\n", elapsed)
    for r in 1:min(o.top, n)
        k = order[r]
        λ = E.values[k]
        v = E.vectors[:, k]
        share = [sum(abs2(v[j]) for j in 1:n if dofs[j][1] == p) for p in eachindex(patches)]
        share ./= sum(share)
        @printf("  |λ| = %.9f  arg %+.3e  rate %+.3e per unit time  patch shares %s\n",
                abs(λ), angle(λ), log(abs(λ)) / (o.nmap * dt),
                join((@sprintf("%.2f", q) for q in share), " "))
    end
    @printf("  eigenvalues above 1 + 1e-9: %d of %d\n",
            count(>(1 + 1e-9), abs.(E.values)), n)
end

main(OPTS)
