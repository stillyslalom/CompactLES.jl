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
[`apply_along!`](@ref) and rarely construct one directly.
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
    clo_first::Vector{Int}     # first point read by clo[j] (1 = edge; ≤ 0 = ghost)
    chi_first::Vector{Int}     # first point of the unmirrored stencil behind chi[j]
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

The local extent `decomp.n_local[dim]` must be at least `2M + 1` for
right-hand-side half-width `M`, plus one more than the closure rows applied at
this rank's closed ends (`2nc + 1` for `nc` rows with both ends closed, `nc + 1`
with one, none on a periodic or interior block), and a closure row's stencil
must fit inside the block and the halo of the opposite end. `M` must not
exceed `decomp.n_halo`. Each condition raises an error when violated.
Construction allocates the packed-line buffer (`n × lines` for `dim == 1`,
`lines × n` otherwise) and factorizes the line solver. Unless the scheme has an
identity left-hand side, a decomposed `dim` makes that factorization collective,
so every rank of the sub-communicator along `dim` must build the plan.

`lo_closures` and `hi_closures` substitute an alternative closure-row set at
the corresponding closed end, in place of `scheme.closures`. This is the
patch-interface hook: [`interface_closures`](@ref) supplies rows whose
right-hand sides read the exchanged ghost data (`ClosureRow.first ≤ 0`), which
`plan_direction` verifies fit inside the halo width. Like the fold keywords,
either may be passed on every rank and acts only on the rank owning that edge.
"""
function plan_direction(decomp::Decomp, scheme::CompactScheme{T}, dim::Int,
                        h::Real; lo_fold::Union{Nothing,Int}=nothing,
                        hi_fold::Union{Nothing,Int}=nothing,
                        lo_closures::Union{Nothing,Vector{ClosureRow{T}}}=nothing,
                        hi_closures::Union{Nothing,Vector{ClosureRow{T}}}=nothing) where {T}
    n = decomp.n_local[dim]
    M = halfwidth(scheme)
    M <= decomp.n_halo || error("stencil half-width $M exceeds halo width $(decomp.n_halo)")
    for rows in (lo_closures, hi_closures)
        rows === nothing && continue
        for row in rows
            reach = 1 - row.first
            reach <= decomp.n_halo || error(
                "closure row of scheme '$(scheme.name)' reads $reach ghost " *
                "layers; halo width is $(decomp.n_halo)")
        end
        # A shortened closure set exposes interior rows whose RHS reaches
        # 1 + length(rows) - M points past the edge; that reach must also fit.
        M - length(rows) <= decomp.n_halo || error(
            "interior stencil behind $(length(rows)) closure rows reads " *
            "$(M - length(rows)) ghost layers; halo width is $(decomp.n_halo)")
    end

    lo_closed = at_lo_edge(decomp, dim) && lo_fold === nothing
    hi_closed = at_hi_edge(decomp, dim) && hi_fold === nothing
    fold_lo = lo_fold !== nothing && at_lo_edge(decomp, dim)
    fold_hi = hi_fold !== nothing && at_hi_edge(decomp, dim)
    lo_rows = lo_closures === nothing ? scheme.closures : lo_closures
    hi_rows = hi_closures === nothing ? scheme.closures : hi_closures
    # Closure rows exist only at a closed end, so the extent a block needs
    # depends on which of its ends are closed: a periodic or interior block
    # needs the interior stencil alone. Counting both ends everywhere forced
    # the six-row T8 set to 13 points on periodic dimensions too.
    nlo = lo_closed ? length(lo_rows) : 0
    nhi = hi_closed ? length(hi_rows) : 0
    need = max(nlo + nhi + 1, 2M + 1)
    n >= need || error(
        "local extent $n along dim $dim too small for scheme '$(scheme.name)' " *
        "(need ≥ $need); use fewer ranks in this dimension")
    # A closure row reads `first + length(rhs) - 1` points in from its edge;
    # past the block that is the opposite end's halo, valid only when that
    # end is open (exchanged, or mirror-filled at a fold).
    reach(rows) = maximum((row.first + length(row.rhs) - 1 for row in rows); init=0)
    for (closed, rows, other_closed) in ((lo_closed, lo_rows, hi_closed),
                                         (hi_closed, hi_rows, lo_closed))
        closed || continue
        limit = n + (other_closed ? 0 : decomp.n_halo)
        reach(rows) <= limit || error(
            "closure rows of scheme '$(scheme.name)' read $(reach(rows)) points " *
            "from a closed end of a block of extent $n along dim $dim; use " *
            "fewer ranks in this dimension")
    end
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
        for (j, row) in enumerate(lo_rows)
            sub, dia, sup = row.lhs
            a[j] = sub; b[j] = dia; c[j] = sup
        end
    end
    if hi_closed
        for (j, row) in enumerate(hi_rows)
            sub, dia, sup = row.lhs
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
    clo = [row.rhs .* scale for row in lo_rows]
    chi = [row.rhs .* (scale * sgn) for row in hi_rows]
    clo_first = [row.first for row in lo_rows]
    chi_first = [row.first for row in hi_rows]

    # An identity left-hand side (α = 0 and every closure row diagonal) makes
    # the fill the answer, so the line solve and its interface stage are both
    # skipped. This is read off the scheme and the supplied closure sets and
    # nothing else: the interface stage carries a collective, and a flag derived
    # from per-rank edge status would produce a deadlock, not an error.
    # `gaussian_filter` is the case in hand.
    identity_lhs(row) = iszero(row.lhs[1]) && isone(row.lhs[2]) && iszero(row.lhs[3])
    explicit = iszero(α) && all(identity_lhs, scheme.closures) &&
               all(identity_lhs, lo_rows) && all(identity_lhs, hi_rows)

    lines = prod(decomp.n_local[k] for k in 1:3 if k != dim)
    line_solver = LineSolver(a, b, c, aL, cR, decomp.sub[dim], decomp.sub_size[dim],
                    decomp.sub_rank[dim], lines; periodic=decomp.periodic[dim],
                    explicit=explicit)
    tr = dim > 1
    DirPlan{T}(dim, n, lines, tr, scheme, line_solver, lo_closed, hi_closed,
               scheme.a0, ci, clo, chi, clo_first, chi_first,
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
    nclo = length(plan.clo)
    nchi = length(plan.chi)
    @threaded plan.lines*n for l in 1:plan.lines
        kk, jj = divrem(l - 1, n1)
        j = jj + 1
        k = kk + 1
        i1, i2 = 1, n
        @inbounds begin
            if plan.lo_closed
                for jr in 1:nclo
                    rhs = plan.clo[jr]
                    i0 = plan.clo_first[jr] - 1
                    acc = zero(T)
                    for κ in eachindex(rhs)
                        acc += rhs[κ] * f[_gidx(Val(D), i0 + κ, j, k, n_halo_d)]
                    end
                    B[jr, l] = acc
                end
                i1 = nclo + 1
            end
            if plan.hi_closed
                for jr in 1:nchi
                    rhs = plan.chi[jr]
                    i0 = n + 2 - plan.chi_first[jr]
                    acc = zero(T)
                    for κ in eachindex(rhs)
                        acc += rhs[κ] * f[_gidx(Val(D), i0 - κ, j, k, n_halo_d)]
                    end
                    B[n + 1 - jr, l] = acc
                end
                i2 = n - nchi
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

# A `nothing` plan slot is reachable in types but not in execution: a plans
# tuple holds `nothing` exactly where the dimension is collapsed or folded, and
# every caller branches on those conditions first. This method makes the dead
# branch statically resolvable, so it neither shows up as a runtime-dispatch
# site in `bench/jetcheck.jl` nor hides a routing bug behind a
# MethodError.
apply_along!(out, ::Nothing, f, decomp::Decomp) =
    error("apply_along! reached a dimension with no plan; the caller should " *
          "have routed a collapsed or folded dimension elsewhere")

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
        # Only dimension d: the pass is a 1-D stencil along d, so the other
        # two dimensions' halos are dead, as in `smooth!`.
        exchange_dim!(f, solver.decomp, d)
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
    plan.lo_closed && jr <= length(plan.clo) && return 1
    plan.hi_closed && jr > plan.n - length(plan.chi) && return 2
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
    # Flattened (jr, kk), allowing a run whose orthogonal dimension is collapsed
    # to divide over the line index. That case is nout = 1, which covers
    # every transverse direction of a planar-2-D grid. Each (kk, jr) writes a
    # distinct nx-long block of B, so the pair is as independent as kk alone was.
    @threaded nout*n*nx for jk in outer_indices(n, nout)
        jr, kk = Tuple(jk)
        @inbounds begin
            kind = _row_kind(plan, jr)
            base = (kk - 1) * nx
            if kind == 1
                rhs = plan.clo[jr]
                i0 = plan.clo_first[jr] - 1
                for i in 1:nx
                    acc = zero(T)
                    for κ in eachindex(rhs)
                        acc += rhs[κ] * (D == 2 ? f[i+o1, i0+κ+o2, kk+o3] :
                                                    f[i+o1, kk+o2, i0+κ+o3])
                    end
                    B[base+i, jr] = acc
                end
            elseif kind == 2
                rhs = plan.chi[n + 1 - jr]
                i0 = n + 2 - plan.chi_first[n + 1 - jr]
                for i in 1:nx
                    acc = zero(T)
                    for κ in eachindex(rhs)
                        acc += rhs[κ] * (D == 2 ? f[i+o1, i0-κ+o2, kk+o3] :
                                                    f[i+o1, kk+o2, i0-κ+o3])
                    end
                    B[base+i, jr] = acc
                end
            elseif sym
                for i in 1:nx
                    acc = a0 * (D == 2 ? f[i+o1, jr+o2, kk+o3] :
                                          f[i+o1, kk+o2, jr+o3])
                    for mm in 1:M
                        acc += ci[mm] * (D == 2 ?
                            (f[i+o1, jr+mm+o2, kk+o3] + f[i+o1, jr-mm+o2, kk+o3]) :
                            (f[i+o1, kk+o2, jr+mm+o3] + f[i+o1, kk+o2, jr-mm+o3]))
                    end
                    B[base+i, jr] = acc
                end
            else
                for i in 1:nx
                    acc = zero(T)
                    for mm in 1:M
                        acc += ci[mm] * (D == 2 ?
                            (f[i+o1, jr+mm+o2, kk+o3] - f[i+o1, jr-mm+o2, kk+o3]) :
                            (f[i+o1, kk+o2, jr+mm+o3] - f[i+o1, kk+o2, jr-mm+o3]))
                    end
                    B[base+i, jr] = acc
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

# --- Fused scatters ---------------------------------------------------------
# The RHS is bandwidth-bound (see bench/phases.jl), and two of its per-solve
# companions were separate full-array passes over data written by the scatter:
# the 1/h gradient rescale and the flux-divergence subtraction into
# dQ. The scatter variants below apply those operations in the scatter itself,
# removing two array streams per line solve. The per-point arithmetic is the
# same product and the same subtraction in the same order as the two-pass
# form, so interior results are bit-identical. Halo cells differ from the
# two-pass rescale, which scaled them along with the interior: gradient-array
# halos hold no data any consumer reads without a fresh exchange, so nothing
# observes the difference.

@inline _jacobian_weight(::Nothing, I, b) = b
@inline _jacobian_weight(inv_J, I, b) = @inbounds inv_J[I] * b

function _scatter_lines_scaled!(out, B::Matrix{T}, plan, decomp::Decomp, scale,
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
            G = _gidx(Val(D), i, j, k, n_halo_d)
            out[G] = B[i, l] * scale[G]
        end
    end
    return out
end

function _scatter_t_scaled!(out, B::Matrix{T}, plan, decomp::Decomp, scale,
                            ::Val{D}) where {T,D}
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
                    G = CartesianIndex(i + o1, jr + o2, kk + o3)
                    out[G] = B[base+i, jr] * scale[G]
                end
            else
                for i in 1:nx
                    G = CartesianIndex(i + o1, kk + o2, jr + o3)
                    out[G] = B[base+i, jr] * scale[G]
                end
            end
        end
    end
    return out
end

function _scatter_lines_subtract!(dQ, c::Int, B::Matrix{T}, plan,
                                  decomp::Decomp, inv_J, ::Val{D}) where {T,D}
    n_halo_d = decomp.n_halo_d
    n = plan.n
    o1, _ = _odims(Val(D))
    n1 = decomp.n_local[o1]
    @threaded plan.lines*n for l in 1:plan.lines
        kk, jj = divrem(l - 1, n1)
        j = jj + 1
        k = kk + 1
        @inbounds for i in 1:n
            G = _gidx(Val(D), i, j, k, n_halo_d)
            dQ[G, c] -= _jacobian_weight(inv_J, G, B[i, l])
        end
    end
    return dQ
end

function _scatter_t_subtract!(dQ, c::Int, B::Matrix{T}, plan, decomp::Decomp,
                              inv_J, ::Val{D}) where {T,D}
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
                    G = CartesianIndex(i + o1, jr + o2, kk + o3)
                    dQ[G, c] -= _jacobian_weight(inv_J, G, B[base+i, jr])
                end
            else
                for i in 1:nx
                    G = CartesianIndex(i + o1, kk + o2, jr + o3)
                    dQ[G, c] -= _jacobian_weight(inv_J, G, B[base+i, jr])
                end
            end
        end
    end
    return dQ
end

apply_along_scaled!(out, ::Nothing, f, decomp::Decomp, scale) =
    error("apply_along_scaled! reached a dimension with no plan; the caller " *
          "should have routed a collapsed or folded dimension elsewhere")

"""
    apply_along_scaled!(out, plan, f, decomp, scale)

[`apply_along!`](@ref) with the scatter multiplying each interior point by
`scale` at the same index: `out[I] = (D f)[I] * scale[I]`. Halo cells of `out`
are left untouched. Same collective and halo contract as `apply_along!`.
"""
function apply_along_scaled!(out, plan::AbstractDirPlan, f, decomp::Decomp,
                             scale)
    d = plan.dim
    if d == 1
        _fill_lines!(plan.B, plan, f, decomp, Val(1))
        solve_lines!(plan.B, plan.line_solver)
        _scatter_lines_scaled!(out, plan.B, plan, decomp, scale, Val(1))
    elseif d == 2
        _fill_t!(plan.B, plan, f, decomp, Val(2))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t_scaled!(out, plan.B, plan, decomp, scale, Val(2))
    else
        _fill_t!(plan.B, plan, f, decomp, Val(3))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t_scaled!(out, plan.B, plan, decomp, scale, Val(3))
    end
    return out
end

apply_along_subtract!(dQ, c::Int, ::Nothing, f, decomp::Decomp, inv_J) =
    error("apply_along_subtract! reached a dimension with no plan; the " *
          "caller should have routed a collapsed or folded dimension elsewhere")

"""
    apply_along_subtract!(dQ, c, plan, f, decomp, inv_J)

[`apply_along!`](@ref) with the scatter subtracting each interior point from
conserved component `c` of `dQ`: `dQ[I, c] -= inv_J[I] * (D f)[I]`, or
`dQ[I, c] -= (D f)[I]` when `inv_J === nothing`. Halo cells of `dQ` are left
untouched. Same collective and halo contract as `apply_along!`.
"""
function apply_along_subtract!(dQ, c::Int, plan::AbstractDirPlan, f,
                               decomp::Decomp, inv_J)
    d = plan.dim
    if d == 1
        _fill_lines!(plan.B, plan, f, decomp, Val(1))
        solve_lines!(plan.B, plan.line_solver)
        _scatter_lines_subtract!(dQ, c, plan.B, plan, decomp, inv_J, Val(1))
    elseif d == 2
        _fill_t!(plan.B, plan, f, decomp, Val(2))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t_subtract!(dQ, c, plan.B, plan, decomp, inv_J, Val(2))
    else
        _fill_t!(plan.B, plan, f, decomp, Val(3))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t_subtract!(dQ, c, plan.B, plan, decomp, inv_J, Val(3))
    end
    return dQ
end
