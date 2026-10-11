"""
    point_spacing(solver, I) -> h

Smallest physical mesh spacing at the padded interior index `I`. Collapsed
directions are excluded. The result includes the local stretch Jacobian and
the metric scale factor, and uses the patch's own spacing, so a refined tile
reports its 3:1 smaller spacing. At a coordinate singularity an active
direction is clamped to the solver's positive floating-point floor, matching
the geometry arrays; `0` is returned only when every direction is collapsed.
"""
@inline point_spacing(solver::SolverLike, I::CartesianIndex{3}) = _point_spacing(solver, I)

# The body of `point_spacing`, for a solver or a `CallbackGeometry`.
@inline function _point_spacing(solver, I::CartesianIndex{3})
    active = solver.decomp.active
    any(active) || return zero(eltype(solver.h))
    x1, m1 = _phys_and_jac(solver, 1, I[1])
    x2, m2 = _phys_and_jac(solver, 2, I[2])
    x3, m3 = _phys_and_jac(solver, 3, I[3])
    s1, s2, s3 = scalefactors(solver.metric, x1, x2, x3)
    scales = (m1 * s1, m2 * s2, m3 * s3)
    T = promote_type(typeof(x1), typeof(x2), typeof(x3), eltype(solver.h))
    floor = positive_floor(T)
    h = T(Inf)
    @inbounds for d in 1:3
        active[d] || continue
        h = min(h, T(solver.h[d]) * max(abs(T(scales[d])), floor))
    end
    return h
end

# What a callback of position reads from a solver or a refined patch: the
# coordinates and the spacing at a padded index (`_domain_coordinate`,
# `_point_spacing`), apart from the solver type. A loop that evaluates a user
# function over a face is then keyed on the function and this, and is not
# compiled again for every tile type that comes to hold the face. `period` is
# the patch level's (`_level_period`), zero on the root, whose coordinates are
# never wrapped.
struct CallbackGeometry{T,St,M}
    origin::NTuple{3,T}
    coord_shift::NTuple{3,T}
    h::NTuple{3,T}
    stretch::St
    metric::M
    decomp::Decomp{T}
    region::BlockRegion
    period::NTuple{3,Int}
end

callback_geometry(s::Solver) =
    CallbackGeometry(s.origin, s.coord_shift, s.h, s.stretch, s.metric, s.decomp,
                     s.region, (0, 0, 0))
function callback_geometry(ps::PatchSolver)
    level = ps.patch.level
    period = level == 0 ? (0, 0, 0) : _level_period(ps.solver, level)
    return CallbackGeometry(ps.origin, ps.coord_shift, ps.h, ps.stretch, ps.metric,
                            ps.decomp, ps.region, period)
end

# The arity decision is made once before a point loop. `applicable` examines
# dispatch only; it does not call user code, so an exception raised by a user
# callback propagates from the actual point unchanged.
struct InitialCallback{Extended,F}
    f::F
end

struct BoundaryCallback{Extended,F}
    f::F
end

@inline function initial_callback(f, x1, x2, x3, h)
    if applicable(f, x1, x2, x3, h)
        return InitialCallback{true,typeof(f)}(f)
    elseif applicable(f, x1, x2, x3)
        return InitialCallback{false,typeof(f)}(f)
    end
    throw(ArgumentError("initial-condition callback must accept (x, y, z) or " *
                        "(x, y, z, h)"))
end

@inline function boundary_callback(f, x1, x2, x3, t, h)
    if applicable(f, x1, x2, x3, t, h)
        return BoundaryCallback{true,typeof(f)}(f)
    elseif applicable(f, x1, x2, x3, t)
        return BoundaryCallback{false,typeof(f)}(f)
    end
    throw(ArgumentError("boundary callback must accept (x, y, z, t) or " *
                        "(x, y, z, t, h)"))
end

@inline pointwise_initial(cb::InitialCallback{false}, x1, x2, x3, h) =
    cb.f(x1, x2, x3)
@inline pointwise_initial(cb::InitialCallback{true}, x1, x2, x3, h) =
    cb.f(x1, x2, x3, h)

@inline pointwise_boundary(cb::BoundaryCallback{false}, x1, x2, x3, t, h) =
    cb.f(x1, x2, x3, t)
@inline pointwise_boundary(cb::BoundaryCallback{true}, x1, x2, x3, t, h) =
    cb.f(x1, x2, x3, t, h)
