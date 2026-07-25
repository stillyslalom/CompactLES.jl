# Directional compact operators.
#
# A DirPlan binds one CompactScheme to one dimension of one decomposition:
# it holds the LineSolver (local Thomas factorization plus the factorized
# reduced interface system), coefficients prescaled by the grid spacing, and a
# scratch matrix B of shape n × lines. Application packs every grid line along
# the dimension into B (threaded), performs the distributed solve, and
# scatters back into the interior of the output field. Input fields must have
# current rank-boundary halos; physical-edge halos are never read.

abstract type AbstractDirPlan end

struct DirPlan{T} <: AbstractDirPlan
    dim::Int
    n::Int                     # local extent along dim
    lines::Int                 # number of lines = product of other local extents
    tr::Bool                   # transposed (lines × n) layout for y/z sweeps
    scheme::CompactScheme{T}
    ls::LineSolver{T}
    lo_closed::Bool            # closure rows at the low edge (fill + matrix)
    hi_closed::Bool
    a0::T                      # center weight (filters)
    ci::Vector{T}              # prescaled interior coefficients
    clo::Vector{Vector{T}}     # prescaled low-edge closure RHS stencils
    chi::Vector{Vector{T}}     # prescaled, sign-adjusted high-edge stencils
    B::Matrix{T}
end

"""
    plan_direction(dec, scheme, dim, h)

Build a DirPlan for `scheme` along dimension `dim` with grid spacing `h`.
"""
function plan_direction(dec::Decomp, scheme::CompactScheme{T}, dim::Int,
                        h::Real; lo_fold::Union{Nothing,Int}=nothing,
                        hi_fold::Union{Nothing,Int}=nothing) where {T}
    n = dec.nloc[dim]
    nc = nclosure(scheme)
    M = halfwidth(scheme)
    n >= max(2nc + 1, 2M + 1) || error(
        "local extent $n along dim $dim too small for scheme '$(scheme.name)' " *
        "(need ≥ $(max(2nc + 1, 2M + 1))); use fewer ranks in this dimension")
    M <= dec.H || error("stencil half-width $M exceeds halo width $(dec.H)")

    lo_closed = at_lo_edge(dec, dim) && lo_fold === nothing
    hi_closed = at_hi_edge(dec, dim) && hi_fold === nothing
    fold_lo = lo_fold !== nothing && at_lo_edge(dec, dim)
    fold_hi = hi_fold !== nothing && at_hi_edge(dec, dim)
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

    lines = prod(dec.nloc[k] for k in 1:3 if k != dim)
    ls = LineSolver(a, b, c, aL, cR, dec.sub[dim], dec.subsize[dim],
                    dec.subrank[dim], lines; periodic=dec.periodic[dim])
    tr = dim > 1
    DirPlan{T}(dim, n, lines, tr, scheme, ls, lo_closed, hi_closed,
               scheme.a0, ci, clo, chi,
               tr ? zeros(T, lines, n) : zeros(T, n, lines))
end

# Map (line coordinate i, orthogonal coordinates j < k in ascending dim order)
# to a halo-offset CartesianIndex, per dimension (per-dim pads Hd).
@inline _gidx(::Val{1}, i, j, k, Hd) = CartesianIndex(i + Hd[1], j + Hd[2], k + Hd[3])
@inline _gidx(::Val{2}, i, j, k, Hd) = CartesianIndex(j + Hd[1], i + Hd[2], k + Hd[3])
@inline _gidx(::Val{3}, i, j, k, Hd) = CartesianIndex(j + Hd[1], k + Hd[2], i + Hd[3])

@inline _odims(::Val{1}) = (2, 3)
@inline _odims(::Val{2}) = (1, 3)
@inline _odims(::Val{3}) = (1, 2)

function _fill_lines!(B::Matrix{T}, pl, f, dec::Decomp,
                      ::Val{D}) where {T,D}
    Hd = dec.Hd
    n = pl.n
    o1, o2 = _odims(Val(D))
    n1 = dec.nloc[o1]
    ci = pl.ci
    M = length(ci)
    a0 = pl.a0
    sym = pl.scheme.symmetric
    nc = length(pl.clo)
    @threaded pl.lines*n for l in 1:pl.lines
        kk, jj = divrem(l - 1, n1)
        j = jj + 1
        k = kk + 1
        i1, i2 = 1, n
        @inbounds begin
            if pl.lo_closed
                for jr in 1:nc
                    rhs = pl.clo[jr]
                    s = zero(T)
                    for κ in eachindex(rhs)
                        s += rhs[κ] * f[_gidx(Val(D), κ, j, k, Hd)]
                    end
                    B[jr, l] = s
                end
                i1 = nc + 1
            end
            if pl.hi_closed
                for jr in 1:nc
                    rhs = pl.chi[jr]
                    s = zero(T)
                    for κ in eachindex(rhs)
                        s += rhs[κ] * f[_gidx(Val(D), n + 1 - κ, j, k, Hd)]
                    end
                    B[n + 1 - jr, l] = s
                end
                i2 = n - nc
            end
            if sym
                for i in i1:i2
                    s = a0 * f[_gidx(Val(D), i, j, k, Hd)]
                    for m in 1:M
                        s += ci[m] * (f[_gidx(Val(D), i + m, j, k, Hd)] +
                                      f[_gidx(Val(D), i - m, j, k, Hd)])
                    end
                    B[i, l] = s
                end
            else
                for i in i1:i2
                    s = zero(T)
                    for m in 1:M
                        s += ci[m] * (f[_gidx(Val(D), i + m, j, k, Hd)] -
                                      f[_gidx(Val(D), i - m, j, k, Hd)])
                    end
                    B[i, l] = s
                end
            end
        end
    end
    return B
end

function _scatter_lines!(out, B::Matrix{T}, pl, dec::Decomp,
                         ::Val{D}) where {T,D}
    Hd = dec.Hd
    n = pl.n
    o1, _ = _odims(Val(D))
    n1 = dec.nloc[o1]
    @threaded pl.lines*n for l in 1:pl.lines
        kk, jj = divrem(l - 1, n1)
        j = jj + 1
        k = kk + 1
        @inbounds for i in 1:n
            out[_gidx(Val(D), i, j, k, Hd)] = B[i, l]
        end
    end
    return out
end

"""
    apply_along!(out, pl, f, dec)

Apply the compact operator of `pl` to field `f` along `pl.dim`, writing the
result into the interior of `out`. `f` must have current rank-boundary halos.
"""
function apply_along!(out, pl::AbstractDirPlan, f, dec::Decomp)
    d = pl.dim
    if d == 1
        _fill_lines!(pl.B, pl, f, dec, Val(1))
        solve_lines!(pl.B, pl.ls)
        _scatter_lines!(out, pl.B, pl, dec, Val(1))
    elseif d == 2
        _fill_t!(pl.B, pl, f, dec, Val(2))
        solve_lines_t!(pl.B, pl.ls)
        _scatter_t!(out, pl.B, pl, dec, Val(2))
    else
        _fill_t!(pl.B, pl, f, dec, Val(3))
        solve_lines_t!(pl.B, pl.ls)
        _scatter_t!(out, pl.B, pl, dec, Val(3))
    end
    return out
end

"""
    filter_field!(f, s; σf=1)

Apply the compact filter along all active dimensions of `f` in place, with a
halo exchange before each directional pass; `σf` is the field's axis parity.
"""
function filter_field!(f, s; σf::Int=1)
    for d in 1:3
        s.dec.active[d] || continue
        exchange_halos!(f, s.dec)
        filt_along!(s.tmpA, f, s, d, σf)
        copy_interior!(f, s.tmpA, s.dec)
    end
    return f
end

# --- Transposed (lines × n) fill and scatter for the y/z sweeps -------------
# Everything iterates in the field's memory order: the innermost index runs
# along x, so both the field reads and the B writes are contiguous. Closure
# rows are classified per row of the sweep dimension.

@inline function _row_kind(pl, jr)
    nc = length(pl.clo)
    pl.lo_closed && jr <= nc && return 1
    pl.hi_closed && jr > pl.n - nc && return 2
    return 0
end

function _fill_t!(B::Matrix{T}, pl, f, dec::Decomp, ::Val{D}) where {T,D}
    Hd = dec.Hd
    n = pl.n
    nx = dec.nloc[1]
    nout = D == 2 ? dec.nloc[3] : dec.nloc[2]   # threaded outer orthogonal dim
    ci = pl.ci
    M = length(ci)
    a0 = pl.a0
    sym = pl.scheme.symmetric
    o1, o2, o3 = Hd
    @threaded nout*n*nx for kk in 1:nout
        @inbounds for jr in 1:n
            kind = _row_kind(pl, jr)
            base = (kk - 1) * nx
            if kind == 1
                rhs = pl.clo[jr]
                for i in 1:nx
                    sacc = zero(T)
                    for κ in eachindex(rhs)
                        sacc += rhs[κ] * (D == 2 ? f[i+o1, κ+o2, kk+o3] :
                                                    f[i+o1, kk+o2, κ+o3])
                    end
                    B[base+i, jr] = sacc
                end
            elseif kind == 2
                rhs = pl.chi[n + 1 - jr]
                for i in 1:nx
                    sacc = zero(T)
                    for κ in eachindex(rhs)
                        sacc += rhs[κ] * (D == 2 ? f[i+o1, n+1-κ+o2, kk+o3] :
                                                    f[i+o1, kk+o2, n+1-κ+o3])
                    end
                    B[base+i, jr] = sacc
                end
            elseif sym
                for i in 1:nx
                    sacc = a0 * (D == 2 ? f[i+o1, jr+o2, kk+o3] :
                                          f[i+o1, kk+o2, jr+o3])
                    for mm in 1:M
                        sacc += ci[mm] * (D == 2 ?
                            (f[i+o1, jr+mm+o2, kk+o3] + f[i+o1, jr-mm+o2, kk+o3]) :
                            (f[i+o1, kk+o2, jr+mm+o3] + f[i+o1, kk+o2, jr-mm+o3]))
                    end
                    B[base+i, jr] = sacc
                end
            else
                for i in 1:nx
                    sacc = zero(T)
                    for mm in 1:M
                        sacc += ci[mm] * (D == 2 ?
                            (f[i+o1, jr+mm+o2, kk+o3] - f[i+o1, jr-mm+o2, kk+o3]) :
                            (f[i+o1, kk+o2, jr+mm+o3] - f[i+o1, kk+o2, jr-mm+o3]))
                    end
                    B[base+i, jr] = sacc
                end
            end
        end
    end
    return B
end

function _scatter_t!(out, B::Matrix{T}, pl, dec::Decomp, ::Val{D}) where {T,D}
    Hd = dec.Hd
    n = pl.n
    nx = dec.nloc[1]
    nout = D == 2 ? dec.nloc[3] : dec.nloc[2]
    o1, o2, o3 = Hd
    @threaded nout*n*nx for kk in 1:nout
        @inbounds for jr in 1:n
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
