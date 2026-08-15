# Directional compact operators.
#
# A DirPlan binds one CompactScheme to one dimension of one decomposition:
# it holds the LineSolver (local Thomas factorization plus the factorized
# reduced interface system), coefficients prescaled by the grid spacing, and a
# scratch matrix B of shape n × lines. Application packs every grid line along
# the dimension into B (threaded), performs the distributed solve, and
# scatters back into the interior of the output field. Input fields must have
# current rank-boundary halos. At a closed physical edge the closure rows
# reference interior points only, so no halo is read there; a fold plan (built
# with `lo_fold`/`hi_fold`) instead runs the interior stencil to the edge and
# reads the mirror halo that `fold_fill!` writes before the sweep.

abstract type AbstractDirPlan end

"""
    DirPlan

Directional execution plan for a tridiagonal [`CompactScheme`](@ref). It binds
the scheme to one dimension and decomposition, with prescaled coefficients,
edge closures, distributed line solver, and reusable packed-line storage.

Plans are constructed by [`setup`](@ref); users normally pass them to
[`apply_along!`](@ref) rather than constructing them directly.
"""
struct DirPlan{T} <: AbstractDirPlan
    dim::Int
    n::Int                     # local extent along dim
    lines::Int                 # number of lines = product of other local extents
    tr::Bool                   # transposed (lines × n) layout for y/z sweeps
    scheme::CompactScheme{T}
    line_solver::LineSolver{T}
    lo_closed::Bool            # closure rows at the low edge (fill + matrix)
    hi_closed::Bool
    a0::T                      # center weight (filters)
    ci::Vector{T}              # prescaled interior coefficients
    clo::Vector{Vector{T}}     # prescaled low-edge closure RHS stencils
    chi::Vector{Vector{T}}     # prescaled, sign-adjusted high-edge stencils
    B::Matrix{T}
end

"""
    plan_direction(decomp, scheme, dim, h; lo_fold=nothing, hi_fold=nothing)

Build a [`DirPlan`](@ref) for `scheme` along dimension `dim` of `decomp`, with
uniform grid spacing `h` along that dimension. Antisymmetric (derivative)
schemes have their right-hand side prescaled by `1/h`; symmetric ones are not
scaled.

`lo_fold` and `hi_fold` carry the antipodal sign `σg = ±1` of a parity fold at
the low and high global edge. Either keyword acts only on the rank owning the
edge in question. The default `nothing` leaves that edge closed, so the scheme's
closure rows are substituted there; a sign instead moves the ghost coupling onto
the diagonal and leaves the right-hand side running the interior stencil, which
reads the mirror halo that `fold_fill!` writes.

The local extent `decomp.n_local[dim]` must be at least `max(2nc + 1, 2M + 1)`
for `nc` closure rows and right-hand-side half-width `M`, and `M` must not
exceed `decomp.n_halo`; both conditions raise an error when violated.
Construction allocates the packed-line buffer (`n × lines` for `dim == 1`,
`lines × n` otherwise) and factorizes the line solver. Unless the scheme has an
identity left-hand side, a decomposed `dim` makes that factorization collective,
so every rank of the sub-communicator along `dim` must build the plan.
"""
function plan_direction(decomp::Decomp, scheme::CompactScheme{T}, dim::Int,
                        h::Real; lo_fold::Union{Nothing,Int}=nothing,
                        hi_fold::Union{Nothing,Int}=nothing) where {T}
    n = decomp.n_local[dim]
    nc = nclosure(scheme)
    M = halfwidth(scheme)
    n >= max(2nc + 1, 2M + 1) || error(
        "local extent $n along dim $dim too small for scheme '$(scheme.name)' " *
        "(need ≥ $(max(2nc + 1, 2M + 1))); use fewer ranks in this dimension")
    M <= decomp.n_halo || error("stencil half-width $M exceeds halo width $(decomp.n_halo)")

    lo_closed = at_lo_edge(decomp, dim) && lo_fold === nothing
    hi_closed = at_hi_edge(decomp, dim) && hi_fold === nothing
    fold_lo = lo_fold !== nothing && at_lo_edge(decomp, dim)
    fold_hi = hi_fold !== nothing && at_hi_edge(decomp, dim)
    α = scheme.alpha
    a = fill(α, n); b = fill(one(T), n); c = fill(α, n)
    if fold_lo
        # Parity fold on a half-offset grid: the LHS coupling to the ghost
        # unknown g₀ = σg·g₁ moves onto the diagonal; the RHS keeps the
        # interior stencil and reads mirror-filled halos (fold_fill!).
        b[1] += T(lo_fold) * α
    end
    if fold_hi
        # Mirror fold at the high end: g_{n+1} = σg·g_n.
        b[n] += T(hi_fold) * α
    end
    if lo_closed
        for j in 1:nc
            sub, dia, sup = scheme.closures[j].lhs
            a[j] = sub; b[j] = dia; c[j] = sup
        end
    end
    if hi_closed
        for j in 1:nc
            sub, dia, sup = scheme.closures[j].lhs
            r = n + 1 - j
            a[r] = sup; b[r] = dia; c[r] = sub   # mirrored
        end
    end
    aL = (lo_closed || fold_lo) ? zero(T) : α
    cR = (hi_closed || fold_hi) ? zero(T) : α

    hinv = one(T) / T(h)
    sgn = scheme.symmetric ? one(T) : -one(T)
    scale = scheme.symmetric ? one(T) : hinv
    ci = scheme.coeffs .* scale
    clo = [row.rhs .* scale for row in scheme.closures]
    chi = [row.rhs .* (scale * sgn) for row in scheme.closures]

    # An identity left-hand side (α = 0 and every closure row diagonal) makes
    # the fill the answer, so the line solve and its interface stage are both
    # skipped. This is read off the scheme and nothing else: the interface
    # stage carries a collective, and a flag derived from per-rank edge status
    # would deadlock rather than fail. `gaussian_filter` is the case in hand.
    explicit = iszero(α) && all(row -> iszero(row.lhs[1]) && isone(row.lhs[2]) &&
                                       iszero(row.lhs[3]), scheme.closures)

    lines = prod(decomp.n_local[k] for k in 1:3 if k != dim)
    line_solver = LineSolver(a, b, c, aL, cR, decomp.sub[dim], decomp.sub_size[dim],
                    decomp.sub_rank[dim], lines; periodic=decomp.periodic[dim],
                    explicit=explicit)
    tr = dim > 1
    DirPlan{T}(dim, n, lines, tr, scheme, line_solver, lo_closed, hi_closed,
               scheme.a0, ci, clo, chi,
               tr ? zeros(T, lines, n) : zeros(T, n, lines))
end

# Map (line coordinate i, orthogonal coordinates j < k in ascending dim order)
# to a halo-offset CartesianIndex, per dimension (`pad` is decomp.n_halo_d).
@inline _gidx(::Val{1}, i, j, k, pad) = CartesianIndex(i + pad[1], j + pad[2], k + pad[3])
@inline _gidx(::Val{2}, i, j, k, pad) = CartesianIndex(j + pad[1], i + pad[2], k + pad[3])
@inline _gidx(::Val{3}, i, j, k, pad) = CartesianIndex(j + pad[1], k + pad[2], i + pad[3])

@inline _odims(::Val{1}) = (2, 3)
@inline _odims(::Val{2}) = (1, 3)
@inline _odims(::Val{3}) = (1, 2)

function _fill_lines!(B::Matrix{T}, plan, f, decomp::Decomp,
                      ::Val{D}) where {T,D}
    n_halo_d = decomp.n_halo_d
    n = plan.n
    o1, o2 = _odims(Val(D))
    n1 = decomp.n_local[o1]
    ci = plan.ci
    M = length(ci)
    a0 = plan.a0
    sym = plan.scheme.symmetric
    nc = length(plan.clo)
    @threaded plan.lines*n for l in 1:plan.lines
        kk, jj = divrem(l - 1, n1)
        j = jj + 1
        k = kk + 1
        i1, i2 = 1, n
        @inbounds begin
            if plan.lo_closed
                for jr in 1:nc
                    rhs = plan.clo[jr]
                    acc = zero(T)
                    for κ in eachindex(rhs)
                        acc += rhs[κ] * f[_gidx(Val(D), κ, j, k, n_halo_d)]
                    end
                    B[jr, l] = acc
                end
                i1 = nc + 1
            end
            if plan.hi_closed
                for jr in 1:nc
                    rhs = plan.chi[jr]
                    acc = zero(T)
                    for κ in eachindex(rhs)
                        acc += rhs[κ] * f[_gidx(Val(D), n + 1 - κ, j, k, n_halo_d)]
                    end
                    B[n + 1 - jr, l] = acc
                end
                i2 = n - nc
            end
            if sym
                for i in i1:i2
                    acc = a0 * f[_gidx(Val(D), i, j, k, n_halo_d)]
                    for m in 1:M
                        acc += ci[m] * (f[_gidx(Val(D), i + m, j, k, n_halo_d)] +
                                        f[_gidx(Val(D), i - m, j, k, n_halo_d)])
                    end
                    B[i, l] = acc
                end
            else
                for i in i1:i2
                    acc = zero(T)
                    for m in 1:M
                        acc += ci[m] * (f[_gidx(Val(D), i + m, j, k, n_halo_d)] -
                                        f[_gidx(Val(D), i - m, j, k, n_halo_d)])
                    end
                    B[i, l] = acc
                end
            end
        end
    end
    return B
end

function _scatter_lines!(out, B::Matrix{T}, plan, decomp::Decomp,
                         ::Val{D}) where {T,D}
    n_halo_d = decomp.n_halo_d
    n = plan.n
    o1, _ = _odims(Val(D))
    n1 = decomp.n_local[o1]
    @threaded plan.lines*n for l in 1:plan.lines
        kk, jj = divrem(l - 1, n1)
        j = jj + 1
        k = kk + 1
        @inbounds for i in 1:n
            out[_gidx(Val(D), i, j, k, n_halo_d)] = B[i, l]
        end
    end
    return out
end

"""
    apply_along!(out, plan, f, decomp)

Apply the compact operator of `plan` to field `f` along `plan.dim`, writing the
result into the interior of `out` and returning `out`. Halo cells of `out` are
left untouched. `f` must have current rank-boundary halos.

`plan.B` is scratch and is overwritten. When `plan.dim` is decomposed the line
solve is collective over the sub-communicator along that dimension, so every
rank of it must call this.
"""
function apply_along!(out, plan::AbstractDirPlan, f, decomp::Decomp)
    d = plan.dim
    if d == 1
        _fill_lines!(plan.B, plan, f, decomp, Val(1))
        solve_lines!(plan.B, plan.line_solver)
        _scatter_lines!(out, plan.B, plan, decomp, Val(1))
    elseif d == 2
        _fill_t!(plan.B, plan, f, decomp, Val(2))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t!(out, plan.B, plan, decomp, Val(2))
    else
        _fill_t!(plan.B, plan, f, decomp, Val(3))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t!(out, plan.B, plan, decomp, Val(3))
    end
    return out
end

"""
    filter_field!(f, solver; σf=1)

Apply the compact filter to `f` in place along every active dimension in
increasing order, exchanging halos before each directional pass, and return `f`.
`σf` is the field's antipodal sign (±1) for the fold on the dimension being
swept; it is ignored on a dimension carrying no fold.

Only the interior of `f` is updated, so its halos hold pre-filter values on
return. `solver.tmp_a` is scratch and is overwritten. Every rank must call this:
the halo exchanges are matched point-to-point pairs and the line solve carries a
collective along each decomposed dimension.
"""
function filter_field!(f, solver; σf::Int=1)
    for d in 1:3
        solver.decomp.active[d] || continue
        exchange_halos!(f, solver.decomp)
        filt_along!(solver.tmp_a, f, solver, d, σf)
        copy_interior!(f, solver.tmp_a, solver.decomp)
    end
    return f
end

# --- Transposed (lines × n) fill and scatter for the y/z sweeps -------------
# Everything iterates in the field's memory order: the innermost index runs
# along x, so both the field reads and the B writes are contiguous. Closure
# rows are classified per row of the sweep dimension.

@inline function _row_kind(plan, jr)
    nc = length(plan.clo)
    plan.lo_closed && jr <= nc && return 1
    plan.hi_closed && jr > plan.n - nc && return 2
    return 0
end

function _fill_t!(B::Matrix{T}, plan, f, decomp::Decomp, ::Val{D}) where {T,D}
    n_halo_d = decomp.n_halo_d
    n = plan.n
    nx = decomp.n_local[1]
    nout = D == 2 ? decomp.n_local[3] : decomp.n_local[2]   # threaded outer orthogonal dim
    ci = plan.ci
    M = length(ci)
    a0 = plan.a0
    sym = plan.scheme.symmetric
    o1, o2, o3 = n_halo_d
    # Flattened (jr, kk) so that a run whose orthogonal dimension is collapsed
    # still divides over the line index. That case is nout = 1, which covers
    # every transverse direction of a planar-2-D grid. Each (kk, jr) writes a
    # distinct nx-long block of B, so the pair is as independent as kk alone was.
    @threaded nout*n*nx for jk in outer_indices(n, nout)
        jr, kk = Tuple(jk)
        @inbounds begin
            kind = _row_kind(plan, jr)
            base = (kk - 1) * nx
            if kind == 1
                rhs = plan.clo[jr]
                for i in 1:nx
                    sensor_sp = zero(T)
                    for κ in eachindex(rhs)
                        sensor_sp += rhs[κ] * (D == 2 ? f[i+o1, κ+o2, kk+o3] :
                                                    f[i+o1, kk+o2, κ+o3])
                    end
                    B[base+i, jr] = sensor_sp
                end
            elseif kind == 2
                rhs = plan.chi[n + 1 - jr]
                for i in 1:nx
                    sensor_sp = zero(T)
                    for κ in eachindex(rhs)
                        sensor_sp += rhs[κ] * (D == 2 ? f[i+o1, n+1-κ+o2, kk+o3] :
                                                    f[i+o1, kk+o2, n+1-κ+o3])
                    end
                    B[base+i, jr] = sensor_sp
                end
            elseif sym
                for i in 1:nx
                    sensor_sp = a0 * (D == 2 ? f[i+o1, jr+o2, kk+o3] :
                                          f[i+o1, kk+o2, jr+o3])
                    for mm in 1:M
                        sensor_sp += ci[mm] * (D == 2 ?
                            (f[i+o1, jr+mm+o2, kk+o3] + f[i+o1, jr-mm+o2, kk+o3]) :
                            (f[i+o1, kk+o2, jr+mm+o3] + f[i+o1, kk+o2, jr-mm+o3]))
                    end
                    B[base+i, jr] = sensor_sp
                end
            else
                for i in 1:nx
                    sensor_sp = zero(T)
                    for mm in 1:M
                        sensor_sp += ci[mm] * (D == 2 ?
                            (f[i+o1, jr+mm+o2, kk+o3] - f[i+o1, jr-mm+o2, kk+o3]) :
                            (f[i+o1, kk+o2, jr+mm+o3] - f[i+o1, kk+o2, jr-mm+o3]))
                    end
                    B[base+i, jr] = sensor_sp
                end
            end
        end
    end
    return B
end

function _scatter_t!(out, B::Matrix{T}, plan, decomp::Decomp, ::Val{D}) where {T,D}
    n_halo_d = decomp.n_halo_d
    n = plan.n
    nx = decomp.n_local[1]
    nout = D == 2 ? decomp.n_local[3] : decomp.n_local[2]
    o1, o2, o3 = n_halo_d
    @threaded nout*n*nx for jk in outer_indices(n, nout)
        jr, kk = Tuple(jk)
        @inbounds begin
            base = (kk - 1) * nx
            if D == 2
                for i in 1:nx
                    out[i+o1, jr+o2, kk+o3] = B[base+i, jr]
                end
            else
                for i in 1:nx
                    out[i+o1, kk+o2, jr+o3] = B[base+i, jr]
                end
            end
        end
    end
    return out
end
