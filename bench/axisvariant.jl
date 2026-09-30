# The patched radial divergences and the near-axis filter of
# bench/axisspectrum.jl and bench/axisrunaway.jl, written as a source term and
# a callback so that the package itself is unchanged. Include after
# CompactLES with `CL` naming the package.

"""
The radial divergence of the mass, radial momentum and energy fluxes rewritten
as a source added to the right-hand side: under `:gradp` the pressure term of
the radial momentum becomes D(p), under `:product` every one of the three
divergences becomes D(F) + F/r. Inviscid fluxes, one species, the radial line
only; a refined run takes the scratch of each patch extent from `cache`.
"""
struct AxisVariant
    mode::Symbol
    cache::Dict{NTuple{3,Int},NTuple{2,Array{Float64,3}}}
end

function AxisVariant(mode::Symbol)
    mode in (:none, :gradp, :product) ||
        error("variant must be none, gradp or product, got $mode")
    return AxisVariant(mode, Dict{NTuple{3,Int},NTuple{2,Array{Float64,3}}}())
end

function CompactLES.add_source!(src::AxisVariant, solver, dQ, Q, t)
    src.mode === :none && return dQ
    eq = solver.equations
    m1, ie = eq.i_mom[1], eq.i_energy
    ir = solver.inv_r
    nx = solver.decomp.n_local[1]
    ρ, u, p = solver.rho, solver.u, solver.p
    f, g = get!(() -> (similar(p), similar(p)), src.cache, size(p))
    # Flux of each component and the sign r·F takes across the axis.
    fluxes = src.mode === :gradp ?
        ((m1, I -> p[I], -1),) :
        ((1, I -> ρ[I] * u[I], 1),
         (m1, I -> ρ[I] * u[I]^2 + p[I], -1),
         (ie, I -> (Q[I, ie] + p[I]) * u[I], 1))
    for (c, F, σ) in fluxes
        # Present: −(1/r)D(r F) (+ p/r for the momentum). Wanted: −D(F) under
        # :gradp for the pressure, −D(F) − F/r under :product.
        for I in CartesianIndices(f)
            f[I] = F(I) / ir[I]
        end
        CL.deriv_along!(g, f, solver, 1, σ)
        for i in 1:nx
            I = padded_index(solver, i, 1, 1)
            dQ[I, c] += ir[I] * g[I]
        end
        for I in CartesianIndices(f)
            f[I] = F(I)
        end
        CL.deriv_along!(g, f, solver, 1, -σ)
        for i in 1:nx
            I = padded_index(solver, i, 1, 1)
            # Under :gradp the metric source p/r stays and cancels this F/r;
            # under :product it is the product rule's.
            dQ[I, c] -= g[I] + ir[I] * f[I]
        end
    end
    return dQ
end

"""
    AxisFilter(far, near, M, filter_cfl, gamma, Q)

The state filter with αf varying along the radial line, run as a callback
after every step on a solver built with `filter_interval = 0`: `far` and
`near` are solvers of the same grid whose `filter_state!` passes are the
filters of the two αf, both at full strength (`filter_cfl = 0`), blended
with weight 1 over the first `M` nodes falling linearly to 0 at node 2M,
and the result relaxed as `filter_weight` relaxes a pass, with the step and
the directional rate (|u| + c)/h of the run.
"""
struct AxisFilter{A,B,S}
    far::A
    near::B
    M::Int
    filter_cfl::Float64
    gamma::Float64
    Qa::S
    Qb::S
end

AxisFilter(far, near, M, filter_cfl, gamma, Q) =
    AxisFilter(far, near, M, filter_cfl, gamma, copy(Q), copy(Q))

function (af::AxisFilter)(solver, Q)
    copyto!(af.Qa, Q)
    copyto!(af.Qb, Q)
    CL.filter_state!(af.far, af.Qa)
    CL.filter_state!(af.near, af.Qb)
    w = 1.0
    if af.filter_cfl > 0
        CL.primitives!(solver, Q)
        rate = 0.0
        for i in 1:solver.decomp.n_local[1]
            I = padded_index(solver, i, 1, 1)
            c = sqrt(af.gamma * max(solver.p[I], 0.0) / solver.rho[I])
            rate = max(rate, (abs(solver.u[I]) + c) / solver.h[1])
        end
        w = min(1.0, solver.dt_prev * rate / af.filter_cfl)
    end
    for c in 1:solver.equations.n_cons, i in 1:solver.decomp.n_local[1]
        I = padded_index(solver, i, 1, 1)
        b = clamp((2af.M - i) / af.M, 0.0, 1.0)
        fa = af.Qa[I, c]
        Q[I, c] += w * (fa - Q[I, c] + b * (af.Qb[I, c] - fa))
    end
    return false
end

CompactLES.rewind!(::AxisFilter, t, step) = nothing
