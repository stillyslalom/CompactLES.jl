# Smooth-evolution accuracy cases for walls, patch interfaces and refinement
# levels, as plain functions with no measurements and no assertions.
#
# `test/convergence.jl` includes this and adds the regression guards;
# `bench/boundaryorder.jl` includes it and runs the full accuracy matrix
# (closures, filters, cadences and the timestep sweep). Keeping the setups
# here means the matrix and the guards cannot drift apart: a guard is set
# from exactly the run the matrix measures.
#
# Every case but `entropy2d_case` is one-dimensional along dimension 1 with
# the transverse dimensions collapsed, so each costs N points rather than N³. Three
# references are used, and each case says which:
#
#   * an exact solution (the entropy wave, the decaying shear mode);
#   * the periodic mirror: a wall problem whose data are even (rho, p) and
#     odd (u) about both walls is the restriction of a periodic problem on
#     the doubled domain, so a periodic run on 2(N − 1) nodes at the same
#     spacing is the wall run without its closure rows. The difference
#     between the two is the closure defect alone;
#   * a fine periodic reference on nested nodes, whose interior error is
#     orders of magnitude below the errors it is compared with.
#
# Errors are reported by region: the wall window (the first and last
# SMOOTH_W nodes of a physical boundary), the interface window (the same at
# a patch or level end), the covered parent nodes under a child level, and
# the interior, each in the maximum norm, together with the composite
# volume-weighted L2 norm that excludes covered parents through the
# package's own masked quadrature. Fitted orders use the actual spacing.
#
# Include after CompactLES.

const SMOOTH_W = 4
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

# --- forward-mode duals for the exact right-hand side ----------------------
#
# The instantaneous truncation error of the assembled right-hand side needs
# the exact divergence of the Navier–Stokes flux of a smooth profile,
# second derivatives included. A two-term dual number, nested once, gives it
# for any profile written in ordinary arithmetic; a hand derivation of the
# energy flux's derivative is longer and easier to get wrong.

struct Dual{T<:Real} <: Real
    v::T
    d::T
end
Dual{T}(x::Real) where {T} = Dual{T}(T(x), zero(T))
Dual{T}(x::Dual) where {T} = Dual{T}(T(x.v), T(x.d))
Base.promote_rule(::Type{Dual{T}}, ::Type{S}) where {T,S<:Real} = Dual{promote_type(T, S)}
Base.zero(::Type{Dual{T}}) where {T} = Dual{T}(zero(T), zero(T))
Base.one(::Type{Dual{T}}) where {T} = Dual{T}(one(T), zero(T))
Base.zero(x::Dual) = zero(typeof(x))
Base.one(x::Dual) = one(typeof(x))
Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.d + b.d)
Base.:-(a::Dual, b::Dual) = Dual(a.v - b.v, a.d - b.d)
Base.:-(a::Dual) = Dual(-a.v, -a.d)
Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.d * b.v + a.v * b.d)
Base.:/(a::Dual, b::Dual) = Dual(a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v))
Base.sin(a::Dual) = Dual(sin(a.v), cos(a.v) * a.d)
Base.cos(a::Dual) = Dual(cos(a.v), -sin(a.v) * a.d)
Base.exp(a::Dual) = Dual(exp(a.v), exp(a.v) * a.d)
Base.:^(a::Dual, p::Integer) = Base.power_by_squaring(a, p)
Base.:^(a::Dual, p::Real) = Dual(a.v^p, p * a.v^(p - 1) * a.d)

"d f / d x at `x`, exactly, for any `f` written in the arithmetic above."
derivative(f, x) = f(Dual(x, one(x))).d

# --- profiles ---------------------------------------------------------------
#
# A profile maps x to the primitive tuple (rho, u, v, p); v is the
# wall-tangential velocity, which the shear mode and the viscous slip wall
# use.

"""
    standing_profile(a, b, c=0; gamma=1.4)

rho = 1 + a cos(πx), u = b sin(πx), v = c cos(πx), isentropic p = rho^γ: even
density, pressure and tangential velocity and odd normal velocity about x = 0
and x = 1, so the wall problem on [0, 1] is the periodic problem on [0, 2) by
symmetry (`mirror_case`). A nonlinear standing wave with no closed form;
a = b = 0.05 stays smooth well past the t = 0.4 the cases run to (the
steepening time is of order 2). The tangential amplitude `c` is zero unless a
slip wall is being measured: a shear traction acts on that component, and a
no-slip wall requires it to vanish at the wall.
"""
standing_profile(a, b, c=0; gamma=1.4) = x -> begin
    rho = 1 + a * cos(pi * x)
    (rho, b * sin(pi * x), c * cos(pi * x), rho^gamma)
end

"""
    entropy_profile(k, phase; u0=0.5, t=0)

rho = 1 + 0.2 sin(k(x − u0 t) + phase) at constant u0 and p = 1, an exact
Euler solution on the periodic [0, 2π), the same wave as `test/patch_tests.jl`
and `test/level_tests.jl`; `k = 3, phase = 0.37` avoids the one favorable
phase of the k = 1 wave and keeps the high-order errors above roundoff.
"""
entropy_profile(k, phase; u0=0.5, t=0.0) =
    x -> (1 + 0.2 * sin(k * (x - u0 * t) + phase), u0 + zero(x), zero(x), one(x))

"""
    shear_profile(V, mu; t=0)

The decaying shear mode v = V sin(πx) exp(−μπ²t) at rho = 1, u = 0, p = 1
between no-slip walls: the tangential stress μ v_x is nonzero at the wall and
the energy equation is exact once `ShearHeatingBalance` removes the viscous
heating μ v_x², so the wall closure differentiates a nontrivial flux and the
solution has a closed form.
"""
shear_profile(V, mu; t=0.0) =
    x -> (one(x), zero(x), V * sin(pi * x) * exp(-mu * pi^2 * t), one(x))

"The energy sink −μ v_x² that keeps `shear_profile` an exact solution."
struct ShearHeatingBalance
    V::Float64
    mu::Float64
end

function CompactLES.add_source!(source::ShearHeatingBalance, solver, dQ, Q, t)
    decay = exp(-2 * source.mu * pi^2 * t)
    ie = solver.equations.i_energy
    for i in 1:solver.decomp.n_local[1]
        x = xcoord(solver, 1, i)
        dQ[gidx(solver, i, 1, 1), ie] -= source.mu * (source.V * pi * cos(pi * x))^2 * decay
    end
    return dQ
end

# --- conserved variables and the exact right-hand side ----------------------

"Conserved tuple of a primitive tuple, in the solver's component order."
function conserved(equations, prim; gamma=1.4)
    rho, u, v, p = prim
    m1, m2, m3 = equations.i_mom
    vals = zeros(typeof(rho), equations.n_cons)
    vals[1] = rho
    vals[m1] = rho * u
    vals[m2] = rho * v
    vals[equations.i_energy] = p / (gamma - 1) + rho * (u * u + v * v) / 2
    return Tuple(vals)
end

"""
    exact_rhs(equations, prof; gamma=1.4, R=1.0, mu=0.0, Pr=0.7, source=nothing)

x → the exact −∂F/∂x of the one-dimensional Navier–Stokes flux of `prof`,
with Stokes' hypothesis (τ = 4μ u_x / 3), conductivity μ c_p / Pr and an
optional energy source, in the solver's component order.
"""
function exact_rhs(equations, prof; gamma=1.4, R=1.0, mu=0.0, Pr=0.7, source=nothing)
    kappa = mu * (gamma * R / (gamma - 1)) / Pr
    function flux(ξ)
        rho, u, v, p = prof(ξ)
        ux = derivative(η -> prof(η)[2], ξ)
        vx = derivative(η -> prof(η)[3], ξ)
        Tx = derivative(η -> (r = prof(η); r[4] / (r[1] * R)), ξ)
        E = p / (gamma - 1) + rho * (u * u + v * v) / 2
        tau = 4 * mu * ux / 3
        (rho * u, rho * u * u + p - tau, rho * u * v - mu * vx,
         (E + p) * u - tau * u - mu * vx * v - kappa * Tx)
    end
    m1, m2, m3 = equations.i_mom
    ie = equations.i_energy
    return x -> begin
        f = ntuple(c -> -derivative(ξ -> flux(ξ)[c], x), 4)
        vals = zeros(typeof(x), equations.n_cons)
        vals[1] = f[1]; vals[m1] = f[2]; vals[m2] = f[3]; vals[ie] = f[4]
        source === nothing || (vals[ie] += source(x))
        Tuple(vals)
    end
end

# --- the cases ----------------------------------------------------------------
#
# Every builder returns `(solver, states)` with the profile's data at t = 0.
# `refine_regions(N)` fixes the refinement's physical endpoints at 5L/12 to
# 7L/12 (and the third level's at 11L/24 to 13L/24) at every N, which the
# region-by-node-count form of the level tests does not: an N/6-node region
# spans (N/6 − 1) cells, so its right end drifts inward by a coarse cell
# per halving of h. N must be a multiple of 24 for the three-level nest.

function refine_regions(N, levels)
    levels == 1 && return nothing
    N % 24 == 0 || error("N = $N: the refinement nest needs N divisible by 24")
    r1 = BlockRegion((5N ÷ 12, 0, 0), (N ÷ 6 + 1, 1, 1))
    levels == 2 && return r1
    # Level-1 node space spans the domain at spacing h/3; 3·offset is the
    # same physical point as the parent's offset.
    r2 = BlockRegion((3 * (5N ÷ 12) + N ÷ 8, 0, 0), (N ÷ 4 + 1, 1, 1))
    return [r1, r2]
end

const SMOOTH_DEFAULTS = (deriv=lele_d1_6(), filt=compact_filter(0.45),
                         filter_interval=0, filter_cfl=0.35, cfl=0.5)

# `precision` selects the solver's element type, and the schemes, transport
# and artificial-property coefficients are converted to it.
function _smooth_solver(n_global, L, bcs, prof; deriv, filt, filter_interval,
                        filter_cfl, cfl, mu=0.0, Pr=0.7, sources=(),
                        precision=Float64, art=ArtParams(enabled=false),
                        kwargs...)
    solver = Solver(n_global=n_global, L_domain=(L, 1.0, 1.0), bcs=bcs,
                    deriv=deriv, filt=filt, filter_interval=filter_interval,
                    filter_cfl=filter_cfl, cfl=cfl, precision=precision,
                    transport=Transport{Float64}(mu0=mu, Pr=Pr), sources=sources,
                    art=art; kwargs...)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> begin
        rho, u, v, p = prof(x)
        Prim(rho=rho, u=(u, v, 0.0), p=p)
    end)
    return solver, states
end

"""
    wall_case(N; viscous=false, slip=!viscous, folded=false, a=0.05, b=0.05,
              c=0.0, mu=0.005, opts...)

The standing wave on [0, 1] with N nodes between adiabatic walls: slip walls
under `slip` and no-slip walls otherwise, with viscosity `mu` when `viscous`
and inviscid otherwise. At `c = 0` the profile's data satisfy both
conditions, since u vanishes and rho, p and T are even at each wall. A nonzero
tangential amplitude `c` adds the component a wall shear traction acts on and
requires a slip wall; with `viscous = true` that is the case the slip wall's
flux contract is measured on.

Under `folded` the walls are `SymmetryPlaneBC` instead: the same physical
domain and the same profile, half a cell offset, so node i sits at (i − ½)/N
and no node lies on either plane. A symmetry plane carries the slip wall's
conditions and not the no-slip ones, so `slip` must hold.
"""
function wall_case(N; viscous=false, slip=!viscous, folded=false, a=0.05, b=0.05,
                   c=0.0, mu=0.005, Pr=0.7, opts...)
    (!folded || slip) || error("a folded wall case is a symmetry plane, which is " *
                               "the slip wall; pass slip = true")
    bc = folded ? SymmetryPlaneBC() : slip ? SlipWallBC() : NoSlipWallBC()
    _smooth_solver((N, 1, 1), 1.0, ((bc, bc), per3[2], per3[3]),
                   standing_profile(a, b, c); mu=viscous ? mu : 0.0, Pr=Pr,
                   merge(SMOOTH_DEFAULTS, opts)...)
end

"""
    mirror_case(N; viscous=false, folded=false, a=0.05, b=0.05, c=0.0,
                mu=0.005, opts...)

The periodic image of `wall_case(N)`: [0, 2) on 2(N − 1) nodes at the same
spacing, so every wall node has a coincident mirror node. Its solution is the
wall problem's, so the difference between the two runs is the closure defect,
derivative and filter rows together, and nothing else. `N` may also be a
fine reference count, nested over the study grids (N − 1 a multiple of each
study's N − 1), for the total error.

Under `folded` it is the image of `wall_case(N; folded = true)` instead: 2N
nodes on [0, 2) at h = 1/N with the origin at h/2, so node j sits at
(j − ½)h and node i of the folded run is node i of this one. A fine folded
reference needs an odd multiple of the study's N, since an even one puts its
nodes half a cell off the study's.
"""
function mirror_case(N; viscous=false, folded=false, a=0.05, b=0.05, c=0.0,
                     mu=0.005, Pr=0.7, opts...)
    n = folded ? 2N : 2(N - 1)
    origin = folded ? (0.5 / N, 0.0, 0.0) : (0.0, 0.0, 0.0)
    _smooth_solver((n, 1, 1), 2.0, per3, standing_profile(a, b, c);
                   origin=origin, mu=viscous ? mu : 0.0, Pr=Pr,
                   merge(SMOOTH_DEFAULTS, opts)...)
end

"""
    shear_case(N; V=0.1, mu=0.005, Twall=NaN, opts...)

The decaying shear mode between adiabatic no-slip walls on [0, 1], exact.
A finite `Twall` makes the walls isothermal; the mode's temperature is the
uniform 1 (rho = p = 1, R = 1), so `Twall = 1` is compatible with it and
exercises the isothermal flux path on the same exact solution.
"""
function shear_case(N; V=0.1, mu=0.005, Pr=0.7, Twall=NaN, opts...)
    bc = NoSlipWallBC(Twall=Twall)
    _smooth_solver((N, 1, 1), 1.0, ((bc, bc), per3[2], per3[3]),
                   shear_profile(V, mu); mu=mu, Pr=Pr,
                   sources=(ShearHeatingBalance(V, mu),),
                   merge(SMOOTH_DEFAULTS, opts)...)
end

"""
    shear_mirror_case(N; V=0.1, mu=0.005, opts...)

The periodic image of `shear_case(N)` on [0, 2), as `mirror_case` is of
`wall_case`: v is odd and rho, p even about both walls, and the heating
balance is periodic, so the difference between the two runs is the
closure defect alone, which the exact solution cannot separate from the
artificial properties once they are on.
"""
function shear_mirror_case(N; V=0.1, mu=0.005, Pr=0.7, opts...)
    _smooth_solver((2(N - 1), 1, 1), 2.0, per3, shear_profile(V, mu); mu=mu, Pr=Pr,
                   sources=(ShearHeatingBalance(V, mu),),
                   merge(SMOOTH_DEFAULTS, opts)...)
end

"""
    entropy_case(N; k=3, phase=0.37, patch_grid=(1, 1, 1), levels=1,
                 subcycle=false, opts...)

The entropy wave on the periodic [0, 2π) with N root nodes, through a
same-level patch interface (`patch_grid = (2, 1, 1)`) or a two- or
three-level nest (`levels`), exact at every time.
"""
function entropy_case(N; k=3, phase=0.37, patch_grid=(1, 1, 1), levels=1,
                      subcycle=false, opts...)
    _smooth_solver((N, 1, 1), 2pi, per3, entropy_profile(k, phase);
                   patch_grid=patch_grid, refine=refine_regions(N, levels),
                   subcycle=subcycle, merge(SMOOTH_DEFAULTS, opts)...)
end

"""
    inflow_case(N; k=2pi, phase=0.37, u0=2.0, opts...)

The entropy wave of `entropy_profile` at the supersonic speed `u0` on [0, 1]
with N nodes, entering through a `DirichletBC` that holds the exact state
at the stage time and leaving through an `NSCBCOutflowBC`, exact at every
time. The inflow data are the only time dependence the boundary carries, so
a temporal study of this case measures how the integrator treats them.
"""
function inflow_case(N; k=2pi, phase=0.37, u0=2.0, opts...)
    inflow = DirichletBC((x, y, z, t) -> begin
        rho, u, _, p = entropy_profile(k, phase; u0=u0, t=t)(x)
        Prim(rho=rho, u=(u, 0.0, 0.0), p=p)
    end)
    _smooth_solver((N, 1, 1), 1.0, ((inflow, NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                   entropy_profile(k, phase; u0=u0); merge(SMOOTH_DEFAULTS, opts)...)
end

"""
    target_inflow_case(N; u0=0.3, period=0.4, eta=2.0, opts...)

A uniform subsonic stream at `u0`, rho = p = 1, on [0, 1] with N nodes,
between an `NSCBCInflowBC` whose pointwise target velocity and temperature
oscillate with `period` and an `NSCBCOutflowBC`. No closed form; the
relaxation rates `eta` are raised from the defaults so that the targets move
the solution within the run.
"""
function target_inflow_case(N; u0=0.3, period=0.4, eta=2.0, opts...)
    target(x, y, z, t) = Prim(u=(u0 * (1 + 0.1 * sinpi(2t / period)), 0.0, 0.0),
                              T_ion=1 + 0.02 * cospi(2t / period), p=1.0, Y=[1.0])
    inflow = NSCBCInflowBC(u=(u0, 0.0, 0.0), T_ion=1.0, Y=[1.0], eta_u=eta,
                           eta_T=eta, target=target)
    _smooth_solver((N, 1, 1), 1.0, ((inflow, NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                   x -> (one(x), u0 + zero(x), zero(x), one(x));
                   merge(SMOOTH_DEFAULTS, opts)...)
end

# --- fixed-step integration and temporal errors -----------------------------

"""
    fixed_step_run!(solver, states, tfinal, nsteps)

Advance to `tfinal` in `nsteps` equal steps through `run!`, one call per
step with the endpoint as the only limit on the step: the solver must be
built with a `cfl` large enough that the endpoint clip always binds. Every
step then takes the path a production step takes, boundary enforcement,
filter and level synchronization included.
"""
function fixed_step_run!(solver, states, tfinal, nsteps)
    dt = tfinal / nsteps
    for n in 1:nsteps
        run!(solver, states; tfinal=n == nsteps ? tfinal : n * dt)
    end
    solver.step == nsteps ||
        error("$(solver.step) steps taken for $nsteps: raise the case's cfl")
    return states
end

"""
    state_difference(solver, a, b; patch=0)

The maximum difference between two states of one solver over every
conserved component and every uncovered interior node, of patch `patch`
alone when it is nonzero.
"""
function state_difference(solver, a, b; patch=0)
    patches = getfield(solver, :patches)
    as = a isa Vector ? a : [a]
    bs = b isa Vector ? b : [b]
    e = 0.0
    for (pi, p) in enumerate(patches)
        patch == 0 || pi == patch || continue
        ps = CompactLES.PatchSolver(solver, p)
        nl = ps.decomp.n_local
        for c in 1:solver.equations.n_cons, k in 1:nl[3], j in 1:nl[2], i in 1:nl[1]
            I = gidx(ps, i, j, k)
            ps.covered[I] == 0 || continue
            e = max(e, abs(as[pi][I, c] - bs[pi][I, c]))
        end
    end
    return e
end

"""
    temporal_errors(build, tfinal, steps, ref_steps; patch=0) -> Vector

The temporal error of `build()` integrated to `tfinal` in each of `steps`
equal steps, against the same case in `ref_steps` steps on the same grid,
so that the spatial error cancels exactly and the difference is the time
integration's alone. `ref_steps` should be a multiple of every entry of
`steps` large enough that the reference's own time error is negligible.
"""
function temporal_errors(build, tfinal, steps, ref_steps; patch=0)
    solver, ref = build()
    fixed_step_run!(solver, ref, tfinal, ref_steps)
    map(steps) do n
        s, q = build()
        fixed_step_run!(s, q, tfinal, n)
        state_difference(s, q, ref; patch=patch)
    end
end

"""
    polynomial_profile()

rho, u and p of degrees 2, 1 and 2 on [0, 1], so that every conserved
variable has degree at most 4 and every inviscid flux component at most 5:
the C6 interior rows and the order-6 level interpolation both reproduce it,
and a right-hand side on its exact data measures an interface row's
polynomial consistency alone. Not a solution; for `rhs_errors` only.
"""
polynomial_profile() =
    x -> (1 + 0.2x - 0.1x * x, 0.3 + 0.2x, zero(x), 1 + 0.1x * x)

"""
    polynomial_case(N; patch_grid=(1, 1, 1), levels=1, opts...)

`polynomial_profile` on N nodes of [0, 1] between `ExtrapolationBC` ends,
through a patch interface or a nest as `entropy_case`; inviscid. The ends'
closure error decays through the compact solve over the tens of nodes to the
interface windows.
"""
function polynomial_case(N; patch_grid=(1, 1, 1), levels=1, opts...)
    bc = ExtrapolationBC()
    _smooth_solver((N, 1, 1), 1.0, ((bc, bc), per3[2], per3[3]),
                   polynomial_profile(); patch_grid=patch_grid,
                   refine=refine_regions(N, levels), merge(SMOOTH_DEFAULTS, opts)...)
end

"""
    viscous_periodic_case(N; a=0.05, b=0.05, mu=0.005, patch_grid=(1, 1, 1),
                          levels=1, subcycle=false, opts...)

The viscous standing wave on the periodic [0, 2) with N root nodes, through
a patch interface or a nest as `entropy_case`; its reference is the
single-patch run on nested fine nodes (`viscous_periodic_case(N_ref)`).
"""
function viscous_periodic_case(N; a=0.05, b=0.05, mu=0.005, Pr=0.7,
                               patch_grid=(1, 1, 1), levels=1, subcycle=false,
                               opts...)
    _smooth_solver((N, 1, 1), 2.0, per3, standing_profile(a, b); mu=mu, Pr=Pr,
                   patch_grid=patch_grid, refine=refine_regions(N, levels),
                   subcycle=subcycle, merge(SMOOTH_DEFAULTS, opts)...)
end

"""
    closed_derivative_errors(N, deriv, f, df; W=SMOOTH_W, folded=false,
                             parity=1) -> (wall, interior, h)

One derivative of `f` on the closed line [0, 1] with N nodes under `deriv`
between slip walls, against `df`, split into the wall window and the
interior, with the grid's own spacing `h` beside them; with a polynomial `f`
of the closure's exactness degree plus one this measures the closure rows'
own pointwise order against the actual spacing 1/(N − 1).

Under `folded` the ends are `SymmetryPlaneBC` instead, no closure row enters,
and `f` must carry `parity` about x = 0 and x = 1, the sign the folded
operator continues it with. The spacing is then 1/N and the window is the
first and last W nodes of a half-offset grid, so a study reads `h` from the
result rather than assuming either convention.
"""
function closed_derivative_errors(N, deriv, f, df; W=SMOOTH_W, folded=false,
                                  parity=1)
    bc = folded ? SymmetryPlaneBC() : SlipWallBC()
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=((bc, bc), per3[2], per3[3]),
                    deriv=deriv, art=ArtParams(enabled=false), filter_interval=0)
    a = CompactLES.field(solver.decomp); da = similar(a)
    for i in 1:N
        a[gidx(solver, i, 1, 1)] = f(xcoord(solver, 1, i))
    end
    CompactLES.deriv_along!(da, a, solver, 1, parity)
    CompactLES._scale_grad!(da, solver, 1)
    wall = interior = 0.0
    for i in 1:N
        e = abs(da[gidx(solver, i, 1, 1)] - df(xcoord(solver, 1, i)))
        if i <= W || i > N - W
            wall = max(wall, e)
        else
            interior = max(interior, e)
        end
    end
    return (wall=wall, interior=interior, h=solver.h[1])
end

# --- references -----------------------------------------------------------------

"""
    NodeReference(solver, Q)

The interior line of a single-patch solution as a callable `x ->` conserved
tuple, for any x on one of its nodes (an error otherwise): the mirror run at
the same spacing or a fine periodic reference on nested nodes.
"""
struct NodeReference
    x0::Float64
    h::Float64
    n::Int
    periodic::Bool
    Q::Matrix{Float64}
end

function NodeReference(solver, Q::AbstractArray)
    n = solver.decomp.n_local[1]
    n == solver.n_global[1] || error("NodeReference needs the whole line on one rank")
    vals = Matrix{Float64}(undef, n, solver.equations.n_cons)
    for c in 1:solver.equations.n_cons, i in 1:n
        vals[i, c] = Q[gidx(solver, i, 1, 1), c]
    end
    NodeReference(xcoord(solver, 1, 1), solver.h[1], n,
                  CompactLES.isperiodic(solver.bcs[1][1]), vals)
end

function (ref::NodeReference)(x)
    g = (x - ref.x0) / ref.h
    gi = round(Int, g)
    abs(g - gi) < 1e-6 || error("x = $x is not a node of the reference grid")
    i = ref.periodic ? mod(gi, ref.n) + 1 : gi + 1
    return ntuple(c -> ref.Q[i, c], size(ref.Q, 2))
end

"x → conserved tuple of the profile, the analytic reference."
analytic_reference(equations, prof; gamma=1.4) =
    x -> conserved(equations, prof(x); gamma=gamma)

# --- regional errors --------------------------------------------------------------

_is_wall(bc) = !(bc isa CompactLES.InterfaceBC) && !CompactLES.isperiodic(bc)

"""
    regional_errors(solver, states, reference; comp=1, W=SMOOTH_W) -> NamedTuple

Errors of component `comp` of `states` against `reference(x)`, split by
region: `wall` (within W nodes of a physical boundary), `interface` (within
W nodes of a patch or level end), `covered` (parent nodes under a child
level), `interior` (the rest), each a maximum norm, plus `l2`, the composite
volume-weighted root-mean-square through the package's masked quadrature,
and `at`, the (patch, node) of the global maximum. A region with no nodes
reads 0.
"""
function regional_errors(solver, states, reference; comp=1, W=SMOOTH_W)
    patches = getfield(solver, :patches)
    multi = states isa Vector
    per_patch = multi ? states : [states]
    wall = interface = covered = interior = 0.0
    at = (0, 0)
    sq = Vector{Array{Float64,3}}(undef, length(patches))
    for (pi, p) in enumerate(patches)
        ps = CompactLES.PatchSolver(solver, p)
        Q = per_patch[pi]
        n = ps.decomp.n_local[1]
        # A patch end carries the physical condition, an `InterfaceBC` toward
        # a neighbor or the parent level, or the periodic wrap of a sole
        # root patch; only the first is a wall, and a sole periodic root has
        # no window at all.
        lo_wall = _is_wall(p.bcs[1][1])
        hi_wall = _is_wall(p.bcs[1][2])
        face = length(patches) > 1
        e2 = zeros(size(Q, 1), size(Q, 2), size(Q, 3))
        for i in 1:n
            I = gidx(ps, i, 1, 1)
            e = abs(Q[I, comp] - reference(xcoord(ps, 1, i))[comp])
            e2[I] = e * e
            if e > max(wall, interface, covered, interior)
                at = (pi, i)
            end
            if ps.covered[I] != 0
                covered = max(covered, e)
            elseif (i <= W && lo_wall) || (i > n - W && hi_wall)
                wall = max(wall, e)
            elseif face && (i <= W || i > n - W)
                interface = max(interface, e)
            else
                interior = max(interior, e)
            end
        end
        sq[pi] = e2
    end
    integral = multi ? volume_integral(solver, sq) : volume_integral(solver, sq[1])
    l2 = sqrt(integral / domain_volume(solver))
    return (wall=wall, interface=interface, covered=covered, interior=interior,
            l2=l2, at=at)
end

"""
    rhs_errors(solver, states, exact; comp=1, W=SMOOTH_W)

The instantaneous right-hand-side truncation error: `compute_rhs!` on the
states as they stand, patches synchronized first, against `exact(x)`, with
the regional split of `regional_errors`.
"""
function rhs_errors(solver, states, exact; comp=1, W=SMOOTH_W)
    if states isa Vector
        dQs = [zero(Q) for Q in states]
        CompactLES._presync!(solver, states)
        for lev in getfield(solver, :levels)
            CompactLES._level_rhs!(solver, lev, states, dQs, false)
        end
        return regional_errors(solver, dQs, exact; comp=comp, W=W)
    end
    dQ = zero(states)
    apply_bcs!(solver, states)
    compute_rhs!(solver, states, dQ)
    return regional_errors(solver, dQ, exact; comp=comp, W=W)
end

# --- a two-dimensional level ------------------------------------------------------
#
# In one dimension a fine patch's boundary plane is a single node coincident
# with a parent node, so the imposed plane is an injection and the
# interpolation reaches the solution only through the ghosts a gradient or a
# filter reads. In two dimensions the plane runs along the other dimension,
# where every third node is interpolated, so the inviscid solution itself
# carries the transfer's order.

"""
    entropy2d_profile(k1, k2, phase; u0=0.5, v0=0.25, t=0)

rho = 1 + 0.2 sin(k1(x − u0 t) + k2(y − v0 t) + phase) at constant velocity
(u0, v0) and p = 1, an exact Euler solution on the periodic [0, 2π)², as a
map (x, y) -> (rho, u, v, p).
"""
entropy2d_profile(k1, k2, phase; u0=0.5, v0=0.25, t=0.0) =
    (x, y) -> (1 + 0.2 * sin(k1 * (x - u0 * t) + k2 * (y - v0 * t) + phase),
               u0 + zero(x), v0 + zero(x), one(x))

"The square level over 5L/12..7L/12 in both dimensions, fixed in physical space."
function refine_region2d(N)
    N % 12 == 0 || error("N = $N: the square level needs N divisible by 12")
    return BlockRegion((5N ÷ 12, 5N ÷ 12, 0), (N ÷ 6 + 1, N ÷ 6 + 1, 1))
end

"""
    entropy2d_case(N; k1=2, k2=1, phase=0.37, subcycle=false, opts...)

The oblique entropy wave on the periodic [0, 2π)² with N² root nodes and one
square refined level, exact at every time.
"""
function entropy2d_case(N; k1=2, k2=1, phase=0.37, subcycle=false, opts...)
    o = merge(SMOOTH_DEFAULTS, opts)
    prof = entropy2d_profile(k1, k2, phase)
    solver = Solver(; n_global=(N, N, 1), L_domain=(2pi, 2pi, 1.0), bcs=per3,
                    art=ArtParams(enabled=false), refine=refine_region2d(N),
                    subcycle=subcycle, o...)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> begin
        rho, u, v, p = prof(x, y)
        Prim(rho=rho, u=(u, v, 0.0), p=p)
    end)
    return solver, states
end

"""
    regional_errors2d(solver, states, reference; comp=1, W=SMOOTH_W)

`regional_errors` for a two-dimensional nest: `reference(x, y)` returns the
conserved tuple; `interface` is the fine patch within W nodes of its
boundary along either dimension, `covered` the parent nodes under it,
`interior` the rest, and `l2` the composite masked root-mean-square.
"""
function regional_errors2d(solver, states, reference; comp=1, W=SMOOTH_W)
    patches = getfield(solver, :patches)
    interface = covered = interior = 0.0
    at = (0, 0)
    sq = Vector{Array{Float64,3}}(undef, length(patches))
    for (pi, p) in enumerate(patches)
        ps = CompactLES.PatchSolver(solver, p)
        Q = states[pi]
        n = ps.decomp.n_local
        e2 = zeros(size(Q, 1), size(Q, 2), size(Q, 3))
        for j in 1:n[2], i in 1:n[1]
            I = gidx(ps, i, j, 1)
            e = abs(Q[I, comp] - reference(xcoord(ps, 1, i), xcoord(ps, 2, j))[comp])
            e2[I] = e * e
            e > max(interface, covered, interior) && (at = (pi, i))
            if ps.covered[I] != 0
                covered = max(covered, e)
            elseif pi > 1 && (min(i, j) <= W || i > n[1] - W || j > n[2] - W)
                interface = max(interface, e)
            else
                interior = max(interior, e)
            end
        end
        sq[pi] = e2
    end
    l2 = sqrt(volume_integral(solver, sq) / domain_volume(solver))
    return (wall=0.0, interface=interface, covered=covered, interior=interior,
            l2=l2, at=at)
end

"The root level's spacing, on a solver of any number of patches."
root_spacing(solver) = getfield(solver, :patches)[1].h[1]

"Least-squares slope of log(err) against log(h): the observed order."
function observed_order(hs, errs)
    x = log.(Float64.(hs)); y = log.(max.(errs, 1e-300))
    n = length(x)
    sx = sum(x); sy = sum(y)
    (n * sum(x .* y) - sx * sy) / (n * sum(x .^ 2) - sx^2)
end

"Order between successive resolutions, actual spacing."
successive_orders(hs, errs) =
    [log(errs[i] / errs[i+1]) / log(hs[i] / hs[i+1]) for i in 1:length(hs)-1]
