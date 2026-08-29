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
    clo_first::Vector{Int}     # always 1: banded closures read from the edge
    chi_first::Vector{Int}
    B::Matrix{T}
end

"""
    plan_direction(decomp, scheme::BandedCompactScheme, dim, h;
                   lo_fold=nothing, hi_fold=nothing)

Build a [`BandPlan`](@ref). Prescaling, fold handling and the `lo_fold` /
`hi_fold` keywords match the [`CompactScheme`](@ref) method. The minimum local
extent is `max(2nc + 1, 2M + 1, 2q + 1)`, because the reduced interface system
couples the last `q` unknowns of one rank to the first `q` of the next.
"""
function plan_direction(decomp::Decomp, scheme::BandedCompactScheme{T}, dim::Int,
                        h::Real; lo_fold::Union{Nothing,Int}=nothing,
                        hi_fold::Union{Nothing,Int}=nothing) where {T}
    n = decomp.n_local[dim]
    q = scheme.q
    nc = nclosure(scheme)
    M = halfwidth(scheme)
    nmin = max(2nc + 1, 2M + 1, 2q + 1)
    n >= nmin || error(
        "local extent $n along dim $dim too small for scheme '$(scheme.name)' " *
        "(need ≥ $nmin); use fewer ranks in this dimension")
    M <= decomp.n_halo || error("stencil half-width $M exceeds halo width $(decomp.n_halo); " *
                        "construct the Solver with n_halo ≥ $M")

    lo_closed = at_lo_edge(decomp, dim) && lo_fold === nothing
    hi_closed = at_hi_edge(decomp, dim) && hi_fold === nothing
    fold_lo = lo_fold !== nothing && at_lo_edge(decomp, dim)
    fold_hi = hi_fold !== nothing && at_hi_edge(decomp, dim)

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
    # silently; both shipped banded schemes satisfy this, a user-supplied one
    # need not.
    (lo_closed || hi_closed) && nc < q &&
        error("scheme '$(scheme.name)' has $nc closure rows but half-bandwidth " *
              "$q; a closed edge needs at least q")
    if lo_closed
        for j in 1:nc
            row = scheme.closures[j].lhs
            for s in -q:q
                j + s < 1 && !iszero(row[q+1+s]) &&
                    error("closure row $j references column $(j+s)")
                Ab[q+1+s, j] = row[q+1+s]
            end
        end
    end
    if hi_closed
        for j in 1:nc
            row = scheme.closures[j].lhs
            r = n + 1 - j
            for s in -q:q
                Ab[q+1+s, r] = row[q+1-s]   # mirrored
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
    clo = [row.rhs .* scale for row in scheme.closures]
    chi = [row.rhs .* (scale * sgn) for row in scheme.closures]

    lines = prod(decomp.n_local[k] for k in 1:3 if k != dim)
    line_solver = BandLineSolver(Ab, AL, CR, decomp.sub[dim], decomp.sub_size[dim],
                        decomp.sub_rank[dim], lines; periodic=decomp.periodic[dim])
    tr = dim > 1
    BandPlan{T}(dim, n, lines, tr, scheme, line_solver, lo_closed, hi_closed,
                scheme.a0, ci, clo, chi, fill(1, nc), fill(1, nc),
                tr ? zeros(T, lines, n) : zeros(T, n, lines))
end
