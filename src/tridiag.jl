# Tridiagonal machinery for compact schemes.
#
# Local systems are solved by a precomputed Thomas factorization. Global
# coupling across MPI ranks (and periodic wrap-around) is handled by the
# spike / reduced-interface method:
#
#   On rank p the global rows read  T x = d − aL x_prev_last e₁ − cR x_next_first eₙ,
#   where T is the local tridiagonal block. With y = T⁻¹d, v = T⁻¹(aL e₁),
#   w = T⁻¹(cR eₙ), the local solution is
#
#       x = y − x_prev_last · v − x_next_first · w.
#
#   Evaluating at the first and last local rows yields, per line, a dense
#   2P × 2P "reduced" system in the interface unknowns
#   z = (x₁⁽⁰⁾, xₙ⁽⁰⁾, x₁⁽¹⁾, xₙ⁽¹⁾, …). The reduced matrix depends only on the
#   scheme, so it is assembled from one Allgather of (v₁, vₙ, w₁, wₙ) and
#   LU-factorized once at plan time. Per application: batched local Thomas
#   solves (threaded over lines), a single Allgather of two interface values
#   per line, one dense triangular solve for all lines at once, and a threaded
#   rank-local correction. Periodic single-rank lines reuse the same path with
#   self-coupling (no communication); non-periodic single-rank lines skip the
#   reduced stage entirely (v = w = 0).

struct TriFactor{T}
    n::Int
    l::Vector{T}      # elimination multipliers (l[1] unused)
    dinv::Vector{T}   # inverses of the modified diagonal
    c::Vector{T}      # (unmodified) super-diagonal
end

function TriFactor(a::Vector{T}, b::Vector{T}, c::Vector{T}) where {T}
    n = length(b)
    l = zeros(T, n)
    dinv = zeros(T, n)
    d = b[1]
    dinv[1] = one(T) / d
    @inbounds for i in 2:n
        l[i] = a[i] * dinv[i-1]
        d = b[i] - l[i] * c[i-1]
        dinv[i] = one(T) / d
    end
    TriFactor{T}(n, l, dinv, copy(c))
end

"In-place Thomas solve of one right-hand side."
function solve_col!(x::AbstractVector{T}, F::TriFactor{T}) where {T}
    n = F.n
    l, dinv, c = F.l, F.dinv, F.c
    @inbounds for i in 2:n
        x[i] -= l[i] * x[i-1]
    end
    @inbounds x[n] *= dinv[n]
    @inbounds for i in (n-1):-1:1
        x[i] = (x[i] - c[i] * x[i+1]) * dinv[i]
    end
    return x
end

mutable struct LineSolver{T}
    n::Int
    F::TriFactor{T}
    v::Vector{T}          # spike from left coupling aL
    w::Vector{T}          # spike from right coupling cR
    hasred::Bool
    red::Any              # LU factorization of the reduced matrix, or nothing
    comm::MPI.Comm        # sub-communicator along the dimension
    P::Int
    p::Int                # 0-based rank within sub-communicator
    lines::Int
    ends::Matrix{T}       # 2 × lines: local (y₁, yₙ) per line
    gath::Array{T,3}      # 2 × lines × P: gathered interface values
    z::Matrix{T}          # 2P × lines: reduced RHS / solution
    zbp::Vector{T}        # contiguous copies of the interface values used by
    zbn::Vector{T}        # the transposed (vectorized) correction sweep
end

"""
    LineSolver(a, b, c, aL, cR, comm, P, p, lines; periodic)

`a`, `b`, `c` are the local sub/diag/super-diagonals (closure rows already
substituted where this rank owns a closed global edge). `aL` is the coupling of
local row 1 to the previous rank's last unknown, `cR` that of local row n to
the next rank's first unknown (zero at closed edges).
"""
function LineSolver(a::Vector{T}, b::Vector{T}, c::Vector{T},
                    aL::T, cR::T, comm::MPI.Comm, P::Int, p::Int,
                    lines::Int; periodic::Bool) where {T}
    F = TriFactor(a, b, c)
    n = length(b)
    v = zeros(T, n)
    w = zeros(T, n)
    if aL != 0
        v[1] = aL
        solve_col!(v, F)
    end
    if cR != 0
        w[n] = cR
        solve_col!(w, F)
    end
    hasred = (P > 1) || periodic
    red = nothing
    if hasred
        quad = T[v[1], v[n], w[1], w[n]]
        allq = zeros(T, 4P)
        if P > 1
            MPI.Allgather!(quad, MPI.UBuffer(allq, 4), comm)
        else
            allq .= quad
        end
        R = zeros(T, 2P, 2P)
        for q in 0:(P-1)
            v1, vn, w1, wn = allq[4q+1], allq[4q+2], allq[4q+3], allq[4q+4]
            rlo, rhi = 2q + 1, 2q + 2
            R[rlo, rlo] += 1
            R[rhi, rhi] += 1
            cprev = 2 * mod(q - 1, P) + 2   # column of x_n on rank q-1
            cnext = 2 * mod(q + 1, P) + 1   # column of x_1 on rank q+1
            R[rlo, cprev] += v1
            R[rlo, cnext] += w1
            R[rhi, cprev] += vn
            R[rhi, cnext] += wn
        end
        red = lu!(R)
    end
    LineSolver{T}(n, F, v, w, hasred, red, comm, P, p, lines,
                  zeros(T, 2, lines), zeros(T, 2, lines, max(P, 1)),
                  zeros(T, 2P, lines), zeros(T, lines), zeros(T, lines))
end

"""
    solve_lines!(B, ls)

Solve the (possibly distributed) tridiagonal system for every column of
`B` (n × lines) in place. MPI collectives run from the serial section only.
"""
function solve_lines!(B::AbstractMatrix{T}, ls::LineSolver{T}) where {T}
    n, L = size(B)
    Threads.@threads for l in 1:L
        solve_col!(view(B, :, l), ls.F)
    end
    ls.hasred || return B

    @inbounds for l in 1:L
        ls.ends[1, l] = B[1, l]
        ls.ends[2, l] = B[n, l]
    end
    if ls.P > 1
        MPI.Allgather!(ls.ends, MPI.UBuffer(vec(ls.gath), 2L), ls.comm)
    else
        copyto!(view(ls.gath, :, :, 1), ls.ends)
    end
    @inbounds for q in 0:(ls.P-1), l in 1:L
        ls.z[2q+1, l] = ls.gath[1, l, q+1]
        ls.z[2q+2, l] = ls.gath[2, l, q+1]
    end
    ldiv!(ls.red, ls.z)

    cprev = 2 * mod(ls.p - 1, ls.P) + 2
    cnext = 2 * mod(ls.p + 1, ls.P) + 1
    v, w, z = ls.v, ls.w, ls.z
    Threads.@threads for l in 1:L
        xl = z[cprev, l]
        xr = z[cnext, l]
        @inbounds for i in 1:n
            B[i, l] -= v[i] * xl + w[i] * xr
        end
    end
    return B
end
