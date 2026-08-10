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
function solve_lines_t!(B::AbstractMatrix{T}, line_solver::LineSolver{T}) where {T}
    line_solver.explicit && return B   # identity LHS: the fill is already the answer
    L, n = size(B)
    F = line_solver.F
    @threaded n*L for rng in collect(_line_chunks(L))
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
    line_solver.hasred || return B

    @inbounds for l in 1:L
        line_solver.ends[1, l] = B[l, 1]
        line_solver.ends[2, l] = B[l, n]
    end
    if line_solver.P > 1
        MPI.Allgather!(line_solver.ends,
                       MPI.UBuffer(vec(line_solver.gath), 2L), line_solver.comm)
    else
        copyto!(view(line_solver.gath, :, :, 1), line_solver.ends)
    end
    @inbounds for q in 0:(line_solver.P-1), l in 1:L
        line_solver.z[2q+1, l] = line_solver.gath[1, l, q+1]
        line_solver.z[2q+2, l] = line_solver.gath[2, l, q+1]
    end
    ldiv!(line_solver.red, line_solver.z)
    cprev = 2 * mod(line_solver.p - 1, line_solver.P) + 2
    cnext = 2 * mod(line_solver.p + 1, line_solver.P) + 1
    zp, zn = line_solver.zbp, line_solver.zbn
    @inbounds for l in 1:L
        zp[l] = line_solver.z[cprev, l]
        zn[l] = line_solver.z[cnext, l]
    end
    v, w = line_solver.v, line_solver.w
    @threaded n*L for rng in collect(_line_chunks(L))
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
function solve_lines_t!(B::AbstractMatrix{T}, line_solver::BandLineSolver{T}) where {T}
    L, n = size(B)
    q = line_solver.q
    F = line_solver.F
    @threaded n*L for rng in collect(_line_chunks(L))
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
    line_solver.hasred || return B

    @inbounds for l in 1:L, r in 1:q
        line_solver.ends[r, l] = B[l, r]
        line_solver.ends[q+r, l] = B[l, n-q+r]
    end
    if line_solver.P > 1
        MPI.Allgather!(line_solver.ends,
                       MPI.UBuffer(vec(line_solver.gath), 2q * L), line_solver.comm)
    else
        copyto!(view(line_solver.gath, :, :, 1), line_solver.ends)
    end
    m2 = 2q
    @inbounds for rk in 0:(line_solver.P-1), l in 1:L, r in 1:m2
        line_solver.z[m2*rk+r, l] = line_solver.gath[r, l, rk+1]
    end
    ldiv!(line_solver.red, line_solver.z)
    cprev = m2 * mod(line_solver.p - 1, line_solver.P) + q
    cnext = m2 * mod(line_solver.p + 1, line_solver.P)
    @inbounds for l in 1:L, t in 1:q
        line_solver.zbp[l, t] = line_solver.z[cprev+t, l]
        line_solver.zbn[l, t] = line_solver.z[cnext+t, l]
    end
    V, W = line_solver.V, line_solver.W
    @threaded n*L for rng in collect(_line_chunks(L))
        isempty(rng) && continue
        @inbounds for i in 1:n, t in 1:q
            vi = V[i, t]
            wi = W[i, t]
            @simd for l in rng
                B[l, i] -= vi * line_solver.zbp[l, t] + wi * line_solver.zbn[l, t]
            end
        end
    end
    return B
end
