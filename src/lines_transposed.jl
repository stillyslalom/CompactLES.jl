# Cache-friendly transposed line path for the y and z sweeps.
#
# The original layout stores each batch of lines as B (n × lines) filled
# line-by-line, which for dims 2 and 3 reads the field with strides of nx or
# nx·ny — the dominant cache cost of the whole RHS. The transposed path
# stores B as (lines × n) and iterates everything in the field's memory
# order: fills and scatters read/write both the field and B contiguously
# (the innermost index runs along x), and the Thomas/banded elimination is
# rewritten as row-outer sweeps vectorized across contiguous line blocks —
# the classic batched-SIMD tridiagonal, threaded by chunking the lines. The
# x sweep keeps the original layout, which is already contiguous both ways.

# --- Vectorized batched solves ---------------------------------------------

function _line_chunks(L::Int)
    nth = max(Threads.nthreads(), 1)
    chunk = cld(L, nth)
    ((1 + (t - 1) * chunk):min(t * chunk, L) for t in 1:nth)
end

"Row-sweep Thomas over B (lines × n), vectorized across lines."
function solve_lines_t!(B::AbstractMatrix{T}, ls::LineSolver{T}) where {T}
    L, n = size(B)
    F = ls.F
    Threads.@threads for rng in collect(_line_chunks(L))
        isempty(rng) && continue
        @inbounds begin
            for i in 2:n
                li = F.l[i]
                @simd for l in rng
                    B[l, i] -= li * B[l, i-1]
                end
            end
            dn = F.dinv[n]
            @simd for l in rng
                B[l, n] *= dn
            end
            for i in (n-1):-1:1
                ci = F.c[i]
                di = F.dinv[i]
                @simd for l in rng
                    B[l, i] = (B[l, i] - ci * B[l, i+1]) * di
                end
            end
        end
    end
    ls.hasred || return B

    @inbounds for l in 1:L
        ls.ends[1, l] = B[l, 1]
        ls.ends[2, l] = B[l, n]
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
    zp, zn = ls.zbp, ls.zbn
    @inbounds for l in 1:L
        zp[l] = ls.z[cprev, l]
        zn[l] = ls.z[cnext, l]
    end
    v, w = ls.v, ls.w
    Threads.@threads for rng in collect(_line_chunks(L))
        isempty(rng) && continue
        @inbounds for i in 1:size(B, 2)
            vi = v[i]
            wi = w[i]
            @simd for l in rng
                B[l, i] -= vi * zp[l] + wi * zn[l]
            end
        end
    end
    return B
end

"Row-sweep banded elimination over B (lines × n), vectorized across lines."
function solve_lines_t!(B::AbstractMatrix{T}, ls::BandLineSolver{T}) where {T}
    L, n = size(B)
    q = ls.q
    F = ls.F
    Threads.@threads for rng in collect(_line_chunks(L))
        isempty(rng) && continue
        @inbounds begin
            for k in 1:(n-1)
                for mrow in 1:min(q, n - k)
                    lm = F.L[mrow, k]
                    iszero(lm) && continue
                    @simd for l in rng
                        B[l, k+mrow] -= lm * B[l, k]
                    end
                end
            end
            for i in n:-1:1
                for t in 1:min(q, n - i)
                    ut = F.U[1+t, i]
                    @simd for l in rng
                        B[l, i] -= ut * B[l, i+t]
                    end
                end
                u0 = inv(F.U[1, i])
                @simd for l in rng
                    B[l, i] *= u0
                end
            end
        end
    end
    ls.hasred || return B

    @inbounds for l in 1:L, r in 1:q
        ls.ends[r, l] = B[l, r]
        ls.ends[q+r, l] = B[l, n-q+r]
    end
    if ls.P > 1
        MPI.Allgather!(ls.ends, MPI.UBuffer(vec(ls.gath), 2q * L), ls.comm)
    else
        copyto!(view(ls.gath, :, :, 1), ls.ends)
    end
    m2 = 2q
    @inbounds for rk in 0:(ls.P-1), l in 1:L, r in 1:m2
        ls.z[m2*rk+r, l] = ls.gath[r, l, rk+1]
    end
    ldiv!(ls.red, ls.z)
    cprev = m2 * mod(ls.p - 1, ls.P) + q
    cnext = m2 * mod(ls.p + 1, ls.P)
    @inbounds for l in 1:L, t in 1:q
        ls.zbp[l, t] = ls.z[cprev+t, l]
        ls.zbn[l, t] = ls.z[cnext+t, l]
    end
    V, W = ls.V, ls.W
    Threads.@threads for rng in collect(_line_chunks(L))
        isempty(rng) && continue
        @inbounds for i in 1:n, t in 1:q
            vi = V[i, t]
            wi = W[i, t]
            @simd for l in rng
                B[l, i] -= vi * ls.zbp[l, t] + wi * ls.zbn[l, t]
            end
        end
    end
    return B
end
