# The spectrum of the one-step map of an unrefined run on the r-z axis,
# linearized about its state: whether the axis amplifies a grid-scale mode.
#
#   julia --project=. -t 1 bench/axisspectrum.jl N=64,128,256 alpha=0,0.45,0.47
#   julia --project=. -t 1 bench/axisspectrum.jl geom=plane alpha=0
#   julia --project=. -t 1 bench/axisspectrum.jl variant=none,areap alpha=0,0.47
#   julia --project=. -t 1 bench/axisspectrum.jl geom=origin variant=none,gradp
#
# --- Why this exists ---------------------------------------------------------
#
# In the area form the radial momentum carries the pressure as
# (1/r)D(r p) − p/r, which equals D(p) only where the discrete derivative obeys
# the product rule. On a mode of wavenumber ω times a slowly varying envelope
# the two differ by (1 − k'(ω)) p/r, k' being the derivative of the scheme's
# modified wavenumber: 0 for resolved modes, 0.56 at ω = 2π/3 and 16/3 at
# ω = π for the C6 rows. The divergence of the mass and energy fluxes then
# pairs with a gradient that is not its adjoint, and the axis can feed energy
# into grid-scale acoustics at a rate of order c/r. The package takes D(p) on
# θ-collapsed r-z and the area form on a resolved θ and at the spherical
# origin, where it is (1/r²)D(r² p) − 2p/r. This script linearizes the
# map from the state after one step to the state `nmap` steps later by central
# differences in ρ, ρu_r (and ρu_θ on a resolved θ) and E at every interior
# node, and prints the eigenvalues of largest modulus with |λ|, log|λ| per step
# and per unit time, the radial node at which the eigenvector peaks, and its
# smooth share ‖(v_i + v_{i+1})/2‖/‖v‖ along r over the components (0 for a
# pure sawtooth, 1 for a constant). The measurements are in
# reference/CALIBRATION_APPENDIX.md under "Grid-scale growth at the r-z axis".
#
# Options, each a comma-separated list where it says so; every combination runs
# in one process:
#   geom     `axis` (AxisBC, CylindricalMetric, θ collapsed), `plane`
#            (SymmetryPlaneBC, Cartesian), `origin` (OriginBC, SphericalMetric,
#            θ and φ collapsed) or `resolved` (AxisBC, CylindricalMetric, θ on
#            `ntheta` nodes over the full circle), all on (0, 2] closed by a
#            slip wall
#   N        radial node counts (list)
#   alpha    filter αf (list); 0 runs no filter
#   base     `rest` (ρ = p = 1) or `pulse` (the converging pulse of
#            `axis_level_case`, advanced to `tbase` before linearizing)
#   cfl      the step as a fraction of h/c₀, c₀ = √1.4 (list)
#   fcfl     `filter_cfl`; 0 filters at full strength every step
#   variant  `none`, the package's form; `areap`, the area form on θ-collapsed
#            r-z; `gradp`, D(p) on a resolved θ or at the origin; `product`,
#            every radial divergence of the mass, radial momentum and energy
#            fluxes of θ-collapsed r-z replaced by D(F) + F/r (list), each
#            through a source (`bench/axisvariant.jl`)
#   mu       the viscosity (Pr = 0.7); 0 by default
#   near     αf over the first `M` nodes, tapering linearly to the run's
#            αf at node 2M (a callback in place of the solver's filter); 0 off
#   nmap     steps per map; `eps` the perturbation; `top` eigenvalues printed;
#            `show` > 0 prints the leading eigenvector over that many nodes
#
# The variants are source terms added after the right-hand side, so they are
# patched builds of the divergence in the sense of the appendix: the package
# is unchanged. They are written for an inviscid rest or pulse state and are
# not a general implementation.
#
# Cost: 2n·nmap steps, n being the node count times the component count, plus
# a dense eigensolve of order n: under a second per case at N = 64 and a few
# seconds at N = 256, after about a minute of compilation.
# Scratch tooling, like everything else in bench/: it prints and asserts
# nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES, Printf, LinearAlgebra
using CompactLES: padded_index, xcoord

const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

const OPTS = CL.script_args(ARGS, (geom="axis", N="128", alpha="0.47", base="rest",
                                   tbase=0.2, cfl="0.9", fcfl=0.0, variant="none",
                                   nmap=1, eps=1e-6, top=3, show=0, near=0.0, M=8,
                                   ntheta=8, mu=0.0))

_list(T, s) = [parse(T, x) for x in split(s, ',')]

include(joinpath(@__DIR__, "axisvariant.jl"))

function build(geom, N, α, fcfl, mode, base, near, M, ntheta, mu)
    geom in ("axis", "plane", "origin", "resolved") ||
        error("geom must be axis, plane, origin or resolved, got $geom")
    prof = base == "rest" ? (r -> (one(r), zero(r), zero(r), one(r))) :
           (r -> begin
               rho = 1 + 0.05 * exp(-((r - 0.5) / 0.1)^2)
               (rho, zero(r), zero(r), rho^1.4)
           end)
    lo = geom == "plane" ? SymmetryPlaneBC() : geom == "origin" ? OriginBC() : AxisBC()
    metric = geom == "plane" ? CartesianMetric() :
             geom == "origin" ? SphericalMetric() : CylindricalMetric()
    filt = α > 0 ? compact_filter(α) : compact_filter(0.45)
    bcs = ((lo, SlipWallBC()), per3[2], per3[3])
    kw = (metric=metric, filt=filt, filter_interval=α > 0 && near == 0 ? 1 : 0,
          filter_cfl=fcfl, cfl=50.0)
    if geom == "resolved"
        # θ over the full circle on `ntheta` nodes; the pressure pulse is
        # axisymmetric, so the base state is too.
        s = merge(SMOOTH_DEFAULTS, kw)
        solver = Solver(n_global=(N, ntheta, 1), L_domain=(2.0, 2π, 1.0), bcs=bcs,
                        metric=metric, deriv=s.deriv, filt=s.filt,
                        filter_interval=s.filter_interval, filter_cfl=s.filter_cfl,
                        cfl=s.cfl, transport=ConstantTransport{Float64}(mu0=mu),
                        art=ArtificialProperties(enabled=false),
                        sources=(AxisVariant(mode),))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> begin
            rho, u, v, p = prof(x)
            Prim(rho=rho, u=(u, v, 0.0), p=p)
        end)
        near > 0 && error("near is written for a radial line")
        return solver, Q, nothing
    end
    # The spherical line's collapsed θ sits on the equator.
    extra = geom == "origin" ? (origin=(0.0, π / 2, 0.0),) : (;)
    solver, Q = _smooth_solver((N, 1, 1), 2.0, bcs, prof; merge(SMOOTH_DEFAULTS, kw)...,
                               sources=(AxisVariant(mode),), mu=mu, extra...)
    near > 0 || return solver, Q, nothing
    geom == "origin" && error("near is written for the r-z axis and the plane")
    # The near-axis filter: two solvers of the same grid lend their filter passes.
    far, _ = _smooth_solver((N, 1, 1), 2.0, bcs, prof;
                            merge(SMOOTH_DEFAULTS, kw, (filter_interval=1, filter_cfl=0.0))...)
    close, _ = _smooth_solver((N, 1, 1), 2.0, bcs, prof;
                              merge(SMOOTH_DEFAULTS, kw, (filt=compact_filter(near),
                                    filter_interval=1, filter_cfl=0.0))...)
    return solver, Q, Callback(EveryStep(1), AxisFilter(far, close, M, fcfl, 1.4, Q))
end

function spectrum(o, geom, N, α, cfl, mode)
    solver, Q, cb = build(geom, N, α, o.fcfl, mode, o.base, o.near, o.M, o.ntheta, o.mu)
    h = solver.h[1]
    dt = cfl * h / sqrt(1.4)
    if o.base == "pulse"
        nb = ceil(Int, o.tbase / dt)
        for k in 1:nb
            run!(solver, Q; tfinal=k * o.tbase / nb, callback=cb)
        end
    end
    run!(solver, Q; tfinal=solver.t + dt, callback=cb)
    base = copy(Q)
    t0, s0 = solver.t, solver.step
    saved = (solver.dt_prev, solver.rate_prev, solver.filter_rate_prev)
    eq = solver.equations
    # Every node of the block, the radial index innermost; ρu_θ joins the
    # components on a resolved θ.
    nθ = solver.decomp.n_local[2]
    comps = nθ > 1 ? (1, eq.i_mom[1], eq.i_mom[2], eq.i_energy) :
                     (1, eq.i_mom[1], eq.i_energy)
    dofs = [(padded_index(solver, i, j, 1), c) for c in comps for j in 1:nθ for i in 1:N]
    n = length(dofs)
    function advance(x)
        copyto!(Q, base)
        for (j, (I, c)) in enumerate(dofs)
            Q[I, c] += x[j]
        end
        solver.t = t0
        solver.step = s0
        solver.dt_prev, solver.rate_prev, solver.filter_rate_prev = saved
        for k in 1:o.nmap
            run!(solver, Q; tfinal=t0 + k * dt, callback=cb)
        end
        return [Q[I, c] for (I, c) in dofs]
    end
    J = zeros(n, n)
    x = zeros(n)
    for j in 1:n
        x[j] = o.eps
        yp = advance(x)
        x[j] = -o.eps
        ym = advance(x)
        x[j] = 0.0
        J[:, j] .= (yp .- ym) ./ (2 * o.eps)
    end
    E = eigen(J)
    order = sortperm(abs.(E.values); rev=true)
    @printf("%s %s, N = %d%s, αf = %s, cfl %.2f, filter weight %.3f, variant %s\n",
            geom, o.base, N, nθ > 1 ? " × $nθ" : "", α > 0 ? string(α) : "off", cfl,
            α > 0 ? CL.filter_weight(solver, 1) : 0.0, mode)
    for r in 1:min(o.top, n)
        k = order[r]
        λ = E.values[k]
        v = E.vectors[:, k]
        a = abs.(v)
        peak = (argmax(a) - 1) % N + 1
        sm = 0.0
        for l in 0:length(comps)*nθ-1, i in 1:N-1
            sm += abs2((v[l*N+i] + v[l*N+i+1]) / 2)
        end
        @printf("  |λ| = %.9f  arg %+.3e  per step %+.3e  per unit time %+.3e  %s\n",
                abs(λ), angle(λ), log(abs(λ)) / o.nmap, log(abs(λ)) / (o.nmap * dt),
                @sprintf("peak node %d  smooth share %.3f", peak, sqrt(sm / sum(abs2, v))))
    end
    if o.show > 0
        # The leading eigenvector over the first `show` nodes, per component,
        # phase-aligned on its largest entry and scaled to it.
        v = E.vectors[:, order[1]]
        v = v ./ v[argmax(abs.(v))]
        names = nθ > 1 ? ("rho", "rho u", "rho v", "E") : ("rho", "rho u", "E")
        for (k, name) in enumerate(names)
            println("    ", rpad(name, 6), join((@sprintf("%+.3f", real(v[(k-1)*N*nθ+i]))
                                          for i in 1:o.show), " "))
        end
    end
    @printf("  eigenvalues above 1 + 1e-9: %d of %d\n",
            count(>(1 + 1e-9), abs.(E.values)), n)
    flush(stdout)
end

function main(o)
    for mode in Symbol.(split(o.variant, ',')), cfl in _list(Float64, o.cfl),
        α in _list(Float64, o.alpha), N in _list(Int, o.N)
        spectrum(o, o.geom, N, α, cfl, mode)
    end
end

main(OPTS)
