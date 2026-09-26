# Initial conditions assembled from regions: shapes carrying states, stacked in
# order and joined by smooth transitions.
#
# A shape is a signed distance, negative inside. A `Layers` initial condition
# converts each shape's distance into a volume fraction through a tanh profile
# and stacks the layers in order, a later layer covering the ones beneath it
# (painter's order), so the fractions at a point always sum to one.
#
# The fractions mix the states as volumes of gas, not as numbers to be
# interpolated one field at a time. Partial densities and momentum are the
# volume-weighted sums of each state's own, so each species' mass and the
# momentum in a transition are those of the volumes that make it up. The
# pressure is the volume-weighted pressure, and the temperature follows from the
# EOS at that density and composition, not from weighting temperatures or
# energies. Two regions at a common pressure therefore stay at that pressure
# through the whole transition, whatever their gases, temperatures and heat
# capacity ratios. Weighting the energy instead would put a pressure defect of
# order (γ₁ − γ₂)/γ in the interface of two gases of different γ, which is
# released as acoustic waves on the first step. For ideal gases at a common
# pressure and temperature the rule reproduces that temperature exactly.

"""
    Shape

A region of space for a [`Layers`](@ref) initial condition, defined by a signed
distance that is negative inside. The provided shapes are [`Slab`](@ref),
[`Box`](@ref), [`Ellipsoid`](@ref), [`Sphere`](@ref), [`Cylinder`](@ref) and
[`LevelSet`](@ref), combined with `∪`, `∩`, `setdiff` and `!`.

Shapes are evaluated in the coordinates of the run's metric: `(x, y, z)` for a
Cartesian run, `(r, θ, z)` for a cylindrical one, `(r, θ, φ)` for a spherical
one. A collapsed direction is ignored, so a `Sphere` in a planar run is a disc
and in a one-dimensional run a segment.
"""
abstract type Shape end

"""
    signed_distance(shape, x, active) -> Float64

The signed distance from point `x` (a three-tuple in the metric's coordinates)
to the boundary of `shape`, negative inside. `active` flags the resolved
directions; a shape ignores the collapsed ones. A new [`Shape`](@ref) subtype
implements this method and nothing else.
"""
function signed_distance end

# The two coordinates other than `dim`, in order, for a bound that is a function
# of them.
_others(x, dim) = dim == 1 ? (x[2], x[3]) : dim == 2 ? (x[1], x[3]) : (x[1], x[2])
_bound_value(b::Real, x, dim) = Float64(b)
_bound_value(f, x, dim) = Float64(f(_others(x, dim)...))

"""
    Slab(dim; lo = -Inf, hi = Inf)

The points whose coordinate `dim` lies between `lo` and `hi`. Either bound may be
a number or a function of the two other coordinates, in order, which makes an
interface of any shape that is single-valued along `dim`:

```julia
Slab(1, hi = 0.1)                                     # x < 0.1
Slab(1, lo = (y, z) -> 0.5 + 0.01cos(2π * y / 0.1))   # beyond a sinusoidal interface
Slab(1, lo = 0.4, hi = 0.45)                          # a layer of gas (a curtain)
```

The distance is measured along `dim`, so a transition width applies along that
axis; across a steep perturbation the transition is thinner than `width` by the
factor `1/sqrt(1 + |∇η|²)`.
"""
struct Slab{L,H} <: Shape
    dim::Int
    lo::L
    hi::H
end

function Slab(dim::Int; lo=-Inf, hi=Inf)
    1 <= dim <= 3 || throw(ArgumentError("Slab: dim must be 1, 2 or 3"))
    return Slab(dim, lo, hi)
end

signed_distance(s::Slab, x, active) =
    max(_bound_value(s.lo, x, s.dim) - x[s.dim], x[s.dim] - _bound_value(s.hi, x, s.dim))

"""
    Box(lo, hi)

The axis-aligned box between corners `lo` and `hi`, each a three-tuple.
"""
struct Box <: Shape
    lo::NTuple{3,Float64}
    hi::NTuple{3,Float64}
    Box(lo, hi) = new(Float64.(Tuple(lo)), Float64.(Tuple(hi)))
end

function signed_distance(b::Box, x, active)
    outside = 0.0
    inside = -Inf
    for d in 1:3
        active[d] || continue
        q = abs(x[d] - (b.lo[d] + b.hi[d]) / 2) - (b.hi[d] - b.lo[d]) / 2
        outside += max(q, 0.0)^2
        inside = max(inside, q)
    end
    return sqrt(outside) + min(inside, 0.0)
end

"""
    Ellipsoid(center, radii)

The ellipsoid with semi-axes `radii` along the coordinate directions about
`center`, both three-tuples. An infinite radius removes that direction, which is
how [`Cylinder`](@ref) is built. The distance is exact for a sphere and scaled by
the smallest semi-axis otherwise.
"""
struct Ellipsoid <: Shape
    center::NTuple{3,Float64}
    radii::NTuple{3,Float64}
    Ellipsoid(center, radii) = new(Float64.(Tuple(center)), Float64.(Tuple(radii)))
end

"""
    Sphere(center, radius)

The ball of `radius` about `center`: a disc in a planar run, a segment in a
one-dimensional one.
"""
Sphere(center, radius::Real) = Ellipsoid(center, (radius, radius, radius))

"""
    Cylinder(center, radius; axis = 3)

The circular cylinder of `radius` about the line through `center` along
coordinate `axis`.
"""
Cylinder(center, radius::Real; axis::Int=3) =
    Ellipsoid(center, ntuple(d -> d == axis ? Inf : Float64(radius), 3))

function signed_distance(e::Ellipsoid, x, active)
    s = 0.0
    a_min = Inf
    for d in 1:3
        (active[d] && isfinite(e.radii[d])) || continue
        s += ((x[d] - e.center[d]) / e.radii[d])^2
        a_min = min(a_min, e.radii[d])
    end
    isfinite(a_min) || return -Inf       # no resolved direction constrains it
    return (sqrt(s) - 1) * a_min
end

"""
    LevelSet(phi)

The region where `phi(x, y, z) < 0`. `phi` should approximate a signed distance
near its zero set, since the transition width is measured in its units.
"""
struct LevelSet{F} <: Shape
    phi::F
end

signed_distance(s::LevelSet, x, active) = Float64(s.phi(x...))

struct ShapeUnion{A,B} <: Shape
    a::A
    b::B
end
struct ShapeIntersection{A,B} <: Shape
    a::A
    b::B
end
struct ShapeComplement{A} <: Shape
    a::A
end

Base.union(a::Shape, b::Shape) = ShapeUnion(a, b)
Base.intersect(a::Shape, b::Shape) = ShapeIntersection(a, b)
Base.:!(a::Shape) = ShapeComplement(a)
Base.setdiff(a::Shape, b::Shape) = ShapeIntersection(a, ShapeComplement(b))

signed_distance(s::ShapeUnion, x, active) =
    min(signed_distance(s.a, x, active), signed_distance(s.b, x, active))
signed_distance(s::ShapeIntersection, x, active) =
    max(signed_distance(s.a, x, active), signed_distance(s.b, x, active))
signed_distance(s::ShapeComplement, x, active) = -signed_distance(s.a, x, active)

"""
    Cells(n)

A transition width of `n` local mesh spacings, for [`Layers`](@ref). A plain
number is a width in the units of the coordinates instead, fixed as the grid is
refined.
"""
struct Cells
    n::Float64
end

_width_value(w::Real, h) = Float64(w)
_width_value(w::Cells, h) = w.n * h

"""
    Layer(shape, state; width = nothing)

One region of a [`Layers`](@ref) initial condition. `shape => state` is the
short form, taking the width of the enclosing `Layers`; `Layer` is needed only
to give this region its own `width`, for example a sharper shock beside a
thicker diffuse interface.
"""
struct Layer{S<:Shape,P,W}
    shape::S
    state::P
    width::W
end

Layer(shape::Shape, state; width=nothing) = Layer(shape, state, width)

_as_layer(l::Layer) = l
_as_layer(p::Pair{<:Shape}) = Layer(p.first, p.second, nothing)
_as_layer(x) = throw(ArgumentError("Layers: a region is `shape => state` or a " *
                                   "Layer, got $(typeof(x))"))

# erf from the series e^{-x²} Σₙ 2ⁿ x^{2n+1} / (2n+1)!!, whose terms are all
# positive, so the sum carries no cancellation and holds to rounding; beyond
# |x| = 6 the function is ±1 in Float64. A transition profile is its only use,
# which does not warrant a SpecialFunctions dependency.
function _erf(x::Float64)
    ax = abs(x)
    ax >= 6 && return copysign(1.0, x)
    term = ax
    s = ax
    n = 0
    while term > eps() * s
        n += 1
        term *= 2ax^2 / (2n + 1)
        s += term
    end
    # The sum rounds up to an ulp past one near |x| = 6.
    return copysign(min(2 / sqrt(π) * exp(-ax^2) * s, 1.0), x)
end

# Volume fraction inside a boundary at signed distance d, transition scale w.
_fraction(profile::Symbol, d, w) =
    w > 0 ? (profile === :erf ? (1 - _erf(d / w)) / 2 : (1 - tanh(d / w)) / 2) :
    Float64(d < 0)

"""
    Layers(background, regions...; width = Cells(3), profile = :tanh)

An initial condition built from regions: `background` everywhere, overlaid in
order by each region, given as `shape => state` or as a [`Layer`](@ref). A later
region covers the earlier ones where they overlap. Pass it to a
[`Problem`](@ref) as `ic`.

Each state is a [`Prim`](@ref), or a function `(x, y, z) -> Prim` for a region
whose state varies in space. The volume fraction of a region across its boundary
is `(1 - tanh(d/width))/2` at signed distance `d`, or `erfc(d/width)/2` with
`profile = :erf`, the diffusion profile many mixing-layer benchmark
specifications state. `width` is the scale of that transition:
[`Cells`](@ref)`(n)` for `n` local mesh spacings, which follows the resolution
and suits a captured shock or a numerically sharp interface, or a number in
coordinate units for a physical diffusion thickness.

The transitions mix the states as volumes of gas: partial densities and
momentum are volume-weighted, the pressure is volume-weighted, and the
temperature comes from the EOS. Regions that share a pressure therefore
share it through the transition, even across gases of different heat capacity
ratio, and each species' mass in a transition is that of the volumes composing
it.

```julia
eos = Nasa9Mixture(["N2", "He", "SF6"])
air = Prim(Y = (1.0, 0.0, 0.0), p = 101_325.0, T_ion = 300.0)
jump = shock_jump(eos, air, 1.22)
ic = Layers(air,
            Slab(1, hi = 0.02) => jump.post,
            Sphere((0.05, 0.0, 0.0), 0.025) =>
                Prim(Y = (0.0, 1.0, 0.0), p = 101_325.0, T_ion = 300.0))
```
"""
struct Layers{B,L,W}
    background::B
    layers::L
    width::W
    profile::Symbol
    # An inner constructor, so that no positional default constructor competes
    # with `Layers(background, regions...)` for a call with two regions.
    Layers{B,L,W}(background, layers, width, profile) where {B,L,W} =
        new{B,L,W}(background, layers, width, profile)
end

function Layers(background, regions...; width=Cells(3), profile::Symbol=:tanh)
    width isa Union{Real,Cells} ||
        throw(ArgumentError("Layers: width is a length or Cells(n)"))
    profile in (:tanh, :erf) ||
        throw(ArgumentError("Layers: profile is :tanh or :erf"))
    layers = map(_as_layer, regions)
    return Layers{typeof(background),typeof(layers),typeof(width)}(background, layers,
                                                                   width, profile)
end

# The form `initialize!` evaluates: the regions bound to the run's EOS and its
# resolved directions, as the four-argument initial condition.
struct BoundLayers{L<:Layers,E}
    layers::L
    eos::E
    active::NTuple{3,Bool}
    n_species::Int
end

_bind_initial(ic::Layers, solver) =
    BoundLayers(ic, solver.eos, solver.decomp.active, nspecies(solver.eos))

_state_at(state::Prim, x) = state
_state_at(state, x) = state(x...)::Prim

# Accumulate `weight` of `state` into (ρY, ρu, p) at point `x`.
function _mix_in(acc, eos, state, x, weight)
    weight > 0 || return acc
    pr = _state_at(state, x)
    ρ, p = _density_pressure(eos, pr)
    ρY, ρu, P = acc
    return (ntuple(k -> ρY[k] + weight * ρ * pr.Y[k], length(ρY)),
            ntuple(d -> ρu[d] + weight * ρ * pr.u[d], 3),
            P + weight * p)
end

function (b::BoundLayers)(x1, x2, x3, h)
    x = (Float64(x1), Float64(x2), Float64(x3))
    layers = b.layers.layers
    # Painter's order: the last layer takes its own fraction, the one beneath it
    # that fraction of what remains, and the background the rest.
    acc = (ntuple(_ -> 0.0, b.n_species), (0.0, 0.0, 0.0), 0.0)
    remaining = 1.0
    for l in reverse(layers)
        remaining > 0 || break
        w = _width_value(l.width === nothing ? b.layers.width : l.width, h)
        d = signed_distance(l.shape, x, b.active)
        α = _fraction(b.layers.profile, d, w)
        acc = _mix_in(acc, b.eos, l.state, x, remaining * α)
        remaining *= 1 - α
    end
    acc = _mix_in(acc, b.eos, b.layers.background, x, remaining)
    ρY, ρu, p = acc
    ρ = sum(ρY)
    return Prim(Y=map(q -> q / ρ, ρY), u=map(q -> q / ρ, ρu), p=p, rho=ρ)
end

# --- Hydrostatic balance ------------------------------------------------------
#
# At rest the momentum right-hand side along the acceleration is −D p + ρ g,
# with D the run's divergence operator along that direction: the compact first
# derivative with its closure rows, prescaled by 1/h. D annihilates constants,
# so D p = ρ g on a line of n nodes has a solution only when ρ g lies in the
# (n − 1)-dimensional range of D. The range misses the direction of D's left
# null vector, an odd-even mode, and a density transition of a few cells has a
# component along it. The balance is therefore imposed at the n − 2 interior
# nodes, where it is exact, and the residual goes to the two end nodes, whose
# normal momentum is reset by a wall condition after every stage. With the
# reference pressure that leaves one free direction z (D z = 0 at the interior
# nodes), chosen to minimize the sum of the squared residuals at the two ends.
#
# The compact filter spreads an end residual into the nodes beside the wall,
# since it does not commute with D at the closure rows, so a filtered run holds
# the balance only to the size of that residual, which decreases as the
# density transition widens. An unfiltered run holds it to round-off.
#
# Each line is solved on its own, so each column of a perturbed interface is
# balanced along the acceleration, and the pressure difference between
# columns drives the instability.

"""
    Hydrostatic(ic; p_ref, at, acceleration = nothing)

An initial condition in discrete hydrostatic balance with the run's body force.
`ic` is a [`Layers`](@ref) initial condition or a function `(x, y, z) -> Prim`
and sets the density, composition and velocity; its pressure is replaced. Pass
the result to a [`Problem`](@ref) as `ic`.

The acceleration is the sum of the run's [`ConstantBodyForce`](@ref) sources,
or `acceleration`, a three-tuple, when given. It must have one nonzero
component, along a resolved direction that is not periodic and carries no
symmetry plane or patch interface. Along each grid line in that direction the
pressure satisfies `D p = ρ g` at every node except the two end nodes, with `D`
the solver's first-derivative operator including its boundary closure rows,
and equals `p_ref` at coordinate `at` along that direction, interpolated
linearly between the neighbouring nodes. The temperature follows from the EOS.
At rest the momentum right-hand side is then at round-off away from the two
end nodes, where a wall condition resets the normal momentum; a continuous
hydrostatic profile leaves the truncation error of `D` instead. The compact
filter does not preserve the balance exactly beside the walls, and the
artificial diffusivities act on a sharp density transition, so a run with
either keeps a small residual motion that decreases as the transition widens.

The balance is computed per grid line, each rank for its own lines, with no
communication. It costs a dense factorization of order `n³` for `n` points
along the acceleration, then `n²` per line.

```julia
g = ConstantBodyForce((0.0, -9.81, 0.0))
ic = Hydrostatic(Layers(light, Slab(2, lo = 0.5) => heavy); p_ref = 1e5, at = 1.0)
prob = Problem(eos = eos, domain = domain, bcs = bcs, ic = ic, sources = (g,))
```
"""
struct Hydrostatic{I}
    ic::I
    p_ref::Float64
    at::Float64
    acceleration::Union{Nothing,NTuple{3,Float64}}
end

function Hydrostatic(ic; p_ref::Real, at::Real, acceleration=nothing)
    acc = acceleration === nothing ? nothing : Float64.(Tuple(acceleration))
    acc === nothing || length(acc) == 3 ||
        throw(ArgumentError("Hydrostatic: acceleration is a three-tuple"))
    return Hydrostatic(ic, Float64(p_ref), Float64(at), acc)
end

# The pressure of each rank-local interior point, and the bound inner
# condition that supplies everything else.
struct BoundHydrostatic{I,P<:AbstractArray{Float64,3}}
    ic::I
    p::P
end

_body_acceleration(sources::Tuple) =
    foldl((acc, s) -> s isa ConstantBodyForce ?
              acc .+ Float64.(s.acceleration) : acc, sources; init=(0.0, 0.0, 0.0))

function _hydrostatic_direction(ic::Hydrostatic, solver)
    g = ic.acceleration === nothing ? _body_acceleration(solver.sources) :
        ic.acceleration
    dims = findall(!iszero, g)
    length(dims) == 1 || throw(ArgumentError(
        "Hydrostatic: the acceleration $g must have exactly one nonzero component"))
    d = dims[1]
    solver.decomp.active[d] || throw(ArgumentError(
        "Hydrostatic: the acceleration is along collapsed direction $d"))
    solver.decomp.periodic[d] && throw(ArgumentError(
        "Hydrostatic: direction $d is periodic, which admits no hydrostatic state"))
    solver.folds[d] === nothing || throw(ArgumentError(
        "Hydrostatic: direction $d carries a fold"))
    any(s -> solver.bcs[d][s] isa InterfaceBC, 1:2) && throw(ArgumentError(
        "Hydrostatic: direction $d ends at a patch interface; the balance " *
        "needs the whole line"))
    (solver.metric isa CartesianMetric && solver.stretch[d] === nothing) ||
        throw(ArgumentError("Hydrostatic: needs a Cartesian metric, unstretched " *
                            "along the acceleration"))
    return d, g[d]
end

# The divergence operator along a line of `n` nodes at spacing `h`, as a dense
# matrix: the scheme's own plan applied to the identity on a one-rank
# decomposition, so the closure rows and the line solve are those of the run.
function _line_operator(scheme, n::Int, h, n_halo::Int, ::Type{T}) where {T}
    decomp = Decomp{T}((n, n, 1), (false, false, false); dims=(1, 1, 1),
                       n_halo=n_halo, comm=MPI.COMM_SELF)
    plan = plan_direction(decomp, scheme, 1, h)
    o1, o2, _ = decomp.n_halo_d
    f = zeros(T, n + 2o1, n + 2o2, 1)
    out = zeros(T, n + 2o1, n + 2o2, 1)
    for j in 1:n
        f[j + o1, j + o2, 1] = one(T)
    end
    apply_along!(out, plan, f, decomp)
    return Float64.(out[o1+1:o1+n, o2+1:o2+n, 1])
end

function _bind_initial(ic::Hydrostatic, solver)
    d, g = _hydrostatic_direction(ic, solver)
    decomp = solver.decomp
    inner = _bind_initial(ic.ic, solver)
    n = decomp.n_global[d]
    goff = solver.region.offset[d]
    xg = [global_xcoord(solver, d, goff + i) for i in 1:n]
    (xg[1] <= ic.at <= xg[n]) || throw(ArgumentError(
        "Hydrostatic: at = $(ic.at) lies outside the grid along direction $d, " *
        "[$(xg[1]), $(xg[n])]"))
    plan = _plan_at(solver.div_plans, d)
    plan = plan isa DevicePlan ? plan.host : plan
    T = eltype(solver.h)
    D = _line_operator(plan.scheme, n, solver.h[d], decomp.n_halo, T)
    # Rows 1:n-2 the interior nodes, row n-1 the low end, row n the reference.
    K = zeros(n, n)
    K[1:n-2, :] .= view(D, 2:n-1, :)
    K[n-1, :] .= view(D, 1, :)
    m = clamp(searchsortedlast(xg, ic.at), 1, n - 1)
    θ = (ic.at - xg[m]) / (xg[m+1] - xg[m])
    K[n, m] = 1 - θ
    K[n, m+1] = θ
    F = lu!(K)
    # The free direction: zero at the interior rows and the reference, a unit
    # residual at the low end, and a residual `cz` at the high end.
    z = zeros(n)
    z[n-1] = 1.0
    ldiv!(F, z)
    cz = dot(view(D, n, :), z)

    o = decomp.n_halo_d
    nl = decomp.n_local
    others = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
    na, nb = nl[others[1]], nl[others[2]]
    P = zeros(nl)
    x0 = (xcoord(solver, 1, 1), xcoord(solver, 2, 1), xcoord(solver, 3, 1))
    cb = initial_callback(inner, x0...,
                          point_spacing(solver, CartesianIndex(o[1] + 1, o[2] + 1,
                                                                 o[3] + 1)))
    R = zeros(n, na)
    col = zeros(n)
    bN = zeros(na)
    for b in 1:nb
        for a in 1:na
            # The line's first local node sets the spacing a `Cells` width
            # reads; unstretched along `d`, it is the same along the line.
            li = ntuple(k -> k == d ? 1 : k == others[1] ? a : b, 3)
            h = point_spacing(solver, CartesianIndex(li[1] + o[1], li[2] + o[2],
                                                     li[3] + o[3]))
            for i in 1:n
                xi = ntuple(k -> k == d ? xg[i] : xcoord(solver, k, li[k]), 3)
                pr = pointwise_initial(cb, xi..., h)
                ρ, _ = _density_pressure(solver.eos, pr)
                # The mixture density as the body force sums it, from the
                # partial densities.
                ρs = 0.0
                for k in eachindex(pr.Y)
                    ρs += ρ * pr.Y[k]
                end
                col[i] = ρs * g
            end
            R[1:n-2, a] .= view(col, 2:n-1)
            R[n-1, a] = col[1]
            R[n, a] = ic.p_ref
            bN[a] = col[n]
        end
        # The lines of one transverse row as one multi-column solve, then the
        # step along z minimizing r₁² + r_n², with r₁ = 0 before it.
        ldiv!(F, R)
        for a in 1:na
            rN = dot(view(D, n, :), view(R, :, a)) - bN[a]
            view(R, :, a) .-= (cz * rN / (1 + cz^2)) .* z
        end
        for a in 1:na, i in 1:nl[d]
            li = ntuple(k -> k == d ? i : k == others[1] ? a : b, 3)
            P[li...] = R[decomp.offset[d] + i, a]
        end
    end
    return BoundHydrostatic(inner, P)
end

function _initialize_interior!(solver::SolverLike, Q, ic::BoundHydrostatic)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    cb = initial_callback(ic.ic, xcoord(solver, 1, 1), xcoord(solver, 2, 1),
                          xcoord(solver, 3, 1),
                          point_spacing(solver, CartesianIndex(o1 + 1, o2 + 1,
                                                                 o3 + 1)))
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        x2 = xcoord(solver, 2, j)
        x3 = xcoord(solver, 3, k)
        for i in 1:nx
            x1 = xcoord(solver, 1, i)
            I = CartesianIndex(i + o1, j + o2, k + o3)
            pr = pointwise_initial(cb, x1, x2, x3, point_spacing(solver, I))
            ρ, _ = _density_pressure(solver.eos, pr)
            write_conserved!(Q, I, solver,
                             Prim{length(pr.Y)}(pr.Y, pr.u, ic.p[i, j, k], NaN, ρ))
        end
    end
    return Q
end

# --- Transitions in time, for boundary targets --------------------------------

"""
    Ramp(eos, from, to; start, duration, speed = nothing)

A boundary target that changes from `from` to `to` over `duration` beginning
at `start`, for [`DirichletBC`](@ref) or the `target` of
[`NSCBCInflowBC`](@ref). Each of `from` and `to` is a [`Prim`](@ref) or a
function `(x, y, z, t) -> Prim`, such as the velocity profile of a jet. Before
`start` the target is `from` exactly and after `start + duration` it is `to`
exactly; in between the two are mixed as volumes of gas, as in
[`Layers`](@ref), with a weight rising as the quintic smooth step
`6s⁵ − 15s⁴ + 10s³`, which has zero first and second derivatives at both ends.

`duration` is a time, or [`Cells`](@ref)`(n)` with `speed`, the speed of the
wave the change launches (the shock speed of [`shock_jump`](@ref)), giving the
time that wave takes to cross `n` local mesh spacings. A shock fired from a
boundary then enters the domain already spread over about `n` cells, as the
width of a [`Layers`](@ref) transition spreads one placed inside it.

A face that injects a jet and later fires a shock is one Dirichlet face whose
target ramps from the jet to the post-shock state:

```julia
eos = Nasa9Mixture(["Air", "SF6"])
jump = shock_jump(eos, air, 1.36; dim = 3, direction = -1)
fire = Ramp(eos, jet, jump.post; start = t_shock, duration = Cells(3),
            speed = jump.shock_speed)
top = DirichletBC(fire)
```

When the condition itself changes at `start`, pair the ramp with a scheduled
[`SwitchableBC`](@ref), such as `SwitchableBC(SlipWallBC(), DirichletBC(fire);
at = t_shock)`, so that the change happens between steps.
"""
struct Ramp{E,A,B,D}
    eos::E
    from::A
    to::B
    start::Float64
    duration::D
    speed::Float64
end

function Ramp(eos, from, to; start::Real, duration, speed=nothing)
    duration isa Union{Real,Cells} ||
        throw(ArgumentError("Ramp: duration is a time or Cells(n)"))
    duration isa Cells && speed === nothing &&
        throw(ArgumentError("Ramp: duration = Cells(n) needs the wave `speed`"))
    duration isa Real && duration < 0 &&
        throw(ArgumentError("Ramp: duration must not be negative"))
    s = speed === nothing ? NaN : abs(Float64(speed))
    duration isa Cells && !(s > 0) &&
        throw(ArgumentError("Ramp: speed must be nonzero"))
    return Ramp(_as_point_eos(eos), from, to, Float64(start), duration, s)
end

_state_at_time(state::Prim, x, t) = state
_state_at_time(state, x, t) = state(x..., t)::Prim

_ramp_duration(d::Real, speed, h) = Float64(d)
_ramp_duration(d::Cells, speed, h) = d.n * h / speed

function (r::Ramp)(x1, x2, x3, t, h)
    x = (Float64(x1), Float64(x2), Float64(x3))
    τ = _ramp_duration(r.duration, r.speed, h)
    s = τ > 0 ? clamp((t - r.start) / τ, 0.0, 1.0) : Float64(t >= r.start)
    s == 0 && return _state_at_time(r.from, x, t)
    s == 1 && return _state_at_time(r.to, x, t)
    α = s^3 * (10 - 15s + 6s^2)
    a = _state_at_time(r.from, x, t)
    b = _state_at_time(r.to, x, t)
    n_species = nspecies(r.eos)
    acc = (ntuple(_ -> 0.0, n_species), (0.0, 0.0, 0.0), 0.0)
    acc = _mix_in(acc, r.eos, a, x, 1 - α)
    acc = _mix_in(acc, r.eos, b, x, α)
    ρY, ρu, p = acc
    ρ = sum(ρY)
    Y = map(q -> q / ρ, ρY)
    # Pressure and temperature, which a characteristic inflow target needs.
    return Prim(Y=Y, u=map(q -> q / ρ, ρu), p=p,
                T_ion=_point_temperature(r.eos, ρ, p, Y))
end
