# The patched radial divergences and the near-axis filter of
# bench/axisspectrum.jl and bench/axisrunaway.jl, written as a source term and
# a callback so that the package itself is unchanged. Include after
# CompactLES with `CL` naming the package.

"""
The radial momentum's pressure term, or every radial divergence, rewritten as
a source added to the right-hand side. The package writes the pressure term of
the radial momentum as ∂p/∂r on θ-collapsed r-z and as the area-weighted
divergence (1/J)D(A p) less the metric source s p elsewhere, s = 1/r under
`CylindricalMetric` and 2/r under `SphericalMetric`. Under `:areap` the
θ-collapsed r-z run takes the area form, the package's before it carried
∂p/∂r; under `:gradp` a run the package gives the area form (the resolved-θ
axis, the spherical origin) takes ∂p/∂r; under `:product` every divergence of
the mass, radial momentum and energy fluxes of a θ-collapsed r-z run becomes
D(F) + F/r. Inviscid fluxes, one species, an unstretched radial dimension; a
refined run takes the scratch of each patch extent from `cache`.
"""
struct AxisVariant
    mode::Symbol
    cache::Dict{NTuple{3,Int},NTuple{2,Array{Float64,3}}}
end

function AxisVariant(mode::Symbol)
    mode in (:none, :areap, :gradp, :product) ||
        error("variant must be none, areap, gradp or product, got $mode")
    return AxisVariant(mode, Dict{NTuple{3,Int},NTuple{2,Array{Float64,3}}}())
end

_interior(solver) = (padded_index(solver, i, j, k)
                     for k in 1:solver.decomp.n_local[3],
                         j in 1:solver.decomp.n_local[2],
                         i in 1:solver.decomp.n_local[1])

function CompactLES.add_source!(src::AxisVariant, solver, dQ, Q, t)
    src.mode === :none && return dQ
    eq = solver.equations
    m1, ie = eq.i_mom[1], eq.i_energy
    gradient = CL._radial_pressure_gradient(solver)
    src.mode === :gradp && gradient &&
        error("variant gradp: this run already takes the pressure term as ∂p/∂r")
    src.mode in (:areap, :product) && !gradient &&
        error("variant $(src.mode) is written for a θ-collapsed r-z run")
    ir, iJ, A = solver.inv_r, solver.inv_J, solver.area_d[1]
    ρ, u, p = solver.rho, solver.u, solver.p
    f, g = get!(() -> (similar(p), similar(p)), src.cache, size(p))
    # Sign of the flux product A·F across the fold, and of F itself.
    σA(c) = solver.folds[1] === nothing ? 1 : solver.folds[1].sigflux[c]
    s = solver.metric isa SphericalMetric ? 2 : 1
    if src.mode in (:areap, :gradp)
        # :gradp adds (1/J)D(A p) − s p/r − D(p); :areap the negative.
        sgn = src.mode === :gradp ? 1 : -1
        f .= A .* p
        CL.deriv_along!(g, f, solver, 1, σA(m1))
        for I in _interior(solver)
            dQ[I, m1] += sgn * (iJ[I] * g[I] - s * ir[I] * p[I])
        end
        copyto!(f, p)
        CL.deriv_along!(g, f, solver, 1, 1)
        for I in _interior(solver)
            dQ[I, m1] -= sgn * g[I]
        end
        return dQ
    end
    # :product. The package's form is −(1/r)D(r F) for the mass and energy
    # and −(1/r)D(r F) − D(p) for the radial momentum, F = ρu² there; the
    # wanted one −D(F) − F/r, which with D(p) is −D(ρu² + p) − ρu²/r.
    fluxes = ((1, I -> ρ[I] * u[I]), (m1, I -> ρ[I] * u[I]^2),
              (ie, I -> (Q[I, ie] + p[I]) * u[I]))
    for (c, F) in fluxes
        σ = σA(c)
        for I in CartesianIndices(f)
            f[I] = F(I) / ir[I]
        end
        CL.deriv_along!(g, f, solver, 1, σ)
        for I in _interior(solver)
            dQ[I, c] += ir[I] * g[I]
        end
        for I in CartesianIndices(f)
            f[I] = F(I)
        end
        CL.deriv_along!(g, f, solver, 1, -σ)
        for I in _interior(solver)
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
