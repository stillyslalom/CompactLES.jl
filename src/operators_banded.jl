# Directional plans for banded compact schemes. BandPlan mirrors DirPlan
# field-for-field (the fill and scatter machinery in operators.jl, both the
# direct and the transposed path, is duck-typed over both), swapping the
# tridiagonal LineSolver for the block-banded BandLineSolver.

"""
    BandPlan

Directional execution plan for a [`BandedCompactScheme`](@ref), using the
distributed banded line solver. It is the pentadiagonal-capable counterpart of
[`DirPlan`](@ref) and is constructed by [`setup`](@ref).
"""
struct BandPlan{T} <: AbstractDirPlan
    dim::Int
    n::Int
    lines::Int
    tr::Bool
    scheme::BandedCompactScheme{T}
    line_solver::BandLineSolver{T}
    lo_closed::Bool
    hi_closed::Bool
    a0::T
    ci::Vector{T}
    clo::Vector{Vector{T}}
    chi::Vector{Vector{T}}
    clo_first::Vector{Int}     # first point read by clo[j] (1 = edge; ≤ 0 = ghost)
    chi_first::Vector{Int}     # first point of the unmirrored stencil behind chi[j]
    B::Matrix{T}
end

"""
    plan_direction(decomp, scheme::BandedCompactScheme, dim, h;
                   lo_fold=nothing, hi_fold=nothing,
                   lo_closures=nothing, hi_closures=nothing)

Build a [`BandPlan`](@ref). Prescaling, fold handling and the `lo_fold` /
`hi_fold` keywords match the [`CompactScheme`](@ref) method, as do
`lo_closures` and `hi_closures`, which substitute a
[`BandedClosureRow`](@ref) set at a closed end in place of the scheme's own
(the patch-interface hook; [`interface_closures`](@ref) supplies rows whose
right-hand sides read the exchanged ghost layers). The minimum local extent
is `max(nlo + nhi + 1, 2M + 1, 2q + 1)` for `nlo` and `nhi` closure rows at
this rank's closed ends, because the reduced interface system couples the
last `q` unknowns of one rank to the first `q` of the next, and a closed end
carries at least `q` closure rows.
"""
function plan_direction(decomp::Decomp, scheme::BandedCompactScheme{T}, dim::Int,
                        h::Real; lo_fold::Union{Nothing,Int}=nothing,
                        hi_fold::Union{Nothing,Int}=nothing,
                        lo_closures::Union{Nothing,Vector{BandedClosureRow{T}}}=nothing,
                        hi_closures::Union{Nothing,Vector{BandedClosureRow{T}}}=nothing,
                        lines_factor::Int=1) where {T}
    n = decomp.n_local[dim]
    q = scheme.q
    M = halfwidth(scheme)
    M <= decomp.n_halo || error("stencil half-width $M exceeds halo width $(decomp.n_halo); " *
                        "construct the Solver with n_halo ≥ $M")
    for rows in (lo_closures, hi_closures)
        rows === nothing && continue
        for row in rows
            reach = 1 - row.first
            reach <= decomp.n_halo || error(
                "closure row of scheme '$(scheme.name)' reads $reach ghost " *
                "layers; halo width is $(decomp.n_halo); construct the Solver " *
                "with n_halo ≥ $reach")
        end
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
    nlo = lo_closed ? length(lo_rows) : 0
    nhi = hi_closed ? length(hi_rows) : 0
    nmin = max(nlo + nhi + 1, 2M + 1, 2q + 1)
    n >= nmin || error(
        "local extent $n along dim $dim too small for scheme '$(scheme.name)' " *
        "(need ≥ $nmin); use fewer ranks in this dimension")
    # A closure row reads `first + length(rhs) - 1` points in from its edge;
    # past the block that is the opposite end's halo, valid only when that
    # end is open.
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

    # Band matrix Ab[q+1+s, i] = A[i, i+s].
    Ab = zeros(T, 2q + 1, n)
    Ab[q+1, :] .= one(T)
    for s in 1:q
        Ab[q+1+s, :] .= scheme.lhs[s]
        Ab[q+1-s, :] .= scheme.lhs[s]
    end
    if fold_lo
        # Parity fold: coupling of row i to ghost column i−m (≤ 0) maps onto
        # interior column m−i+1 with sign σg (half-offset mirror).
        σ = T(lo_fold)
        for i in 1:q, m in 1:q
            i - m >= 1 && continue
            jcol = m - i + 1
            Ab[q + 1 + (jcol - i), i] += σ * scheme.lhs[m]
        end
    end
    if fold_hi
        # High-end fold: coupling of row i to ghost column i+m (> n) maps
        # onto interior column 2n+1−(i+m) with sign σg.
        σ = T(hi_fold)
        for i in (n-q+1):n, m in 1:q
            i + m <= n && continue
            jcol = 2n + 1 - (i + m)
            Ab[q + 1 + (jcol - i), i] += σ * scheme.lhs[m]
        end
    end
    # Fewer closure rows than the half-bandwidth would leave rows nc+1..q
    # coupled to ghost unknowns that the zeroed coupling blocks below drop
    # silently; both built-in banded schemes and the interface rows satisfy
    # this, a user-supplied set need not.
    for (closed, rows) in ((lo_closed, lo_rows), (hi_closed, hi_rows))
        closed && length(rows) < q &&
            error("scheme '$(scheme.name)' has $(length(rows)) closure rows but " *
                  "half-bandwidth $q; a closed edge needs at least q")
    end
    if lo_closed
        for (j, row) in enumerate(lo_rows)
            for s in -q:q
                j + s < 1 && !iszero(row.lhs[q+1+s]) &&
                    error("closure row $j references column $(j+s)")
                Ab[q+1+s, j] = row.lhs[q+1+s]
            end
        end
    end
    if hi_closed
        for (j, row) in enumerate(hi_rows)
            r = n + 1 - j
            for s in -q:q
                Ab[q+1+s, r] = row.lhs[q+1-s]   # mirrored
            end
        end
    end

    # Coupling blocks: row r ≤ q couples to the previous rank's tail element t
    # (t = q ↔ its last unknown) at distance q + r − t, so
    # AL[r, t] = lhs[q + r − t] for t ≥ r; mirrored for CR.
    AL = zeros(T, q, q)
    CR = zeros(T, q, q)
    if !(lo_closed || fold_lo)
        for r in 1:q, t in r:q
            AL[r, t] = scheme.lhs[q+r-t]
        end
    end
    if !(hi_closed || fold_hi)
        for r in 1:q, t in 1:r
            CR[r, t] = scheme.lhs[t+q-r]
        end
    end

    hinv = one(T) / T(h)
    sgn = scheme.symmetric ? one(T) : -one(T)
    scale = scheme.symmetric ? one(T) : hinv
    ci = scheme.coeffs .* scale
    clo = [row.rhs .* scale for row in lo_rows]
    chi = [row.rhs .* (scale * sgn) for row in hi_rows]
    clo_first = [row.first for row in lo_rows]
    chi_first = [row.first for row in hi_rows]

    # `lines_factor` as in the tridiagonal plan: the batched device solve of a
    # stacked level sizes the interface stage for every tile's lines at once.
    lines = prod(decomp.n_local[k] for k in 1:3 if k != dim) * lines_factor
    line_solver = BandLineSolver(Ab, AL, CR, decomp.sub[dim], decomp.sub_size[dim],
                        decomp.sub_rank[dim], lines; periodic=decomp.periodic[dim])
    tr = dim > 1
    BandPlan{T}(dim, n, lines, tr, scheme, line_solver, lo_closed, hi_closed,
                scheme.a0, ci, clo, chi, clo_first, chi_first,
                tr ? zeros(T, lines, n) : zeros(T, n, lines))
end
