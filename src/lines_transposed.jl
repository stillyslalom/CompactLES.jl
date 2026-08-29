# Cache-friendly transposed line path for the y and z sweeps.
#
# The original layout stores each batch of lines as B (n × lines) filled
# line-by-line, which for dims 2 and 3 reads the field with strides of nx or
# nx·ny, the dominant cache cost of the whole RHS. The transposed path
# stores B as (lines × n) and iterates everything in the field's memory
# order: fills and scatters read/write both the field and B contiguously
# (the innermost index runs along x), and the Thomas/banded elimination is
# rewritten as row-outer sweeps vectorized across contiguous line blocks,
# the classic batched-SIMD tridiagonal, threaded by chunking the lines. The
# x sweep keeps the original layout, which is already contiguous both ways.

# --- Vectorized batched solves ---------------------------------------------

# Contiguous blocks of lines, one per thread, as the iteration space of the
# solves below. `Threads.@threads` indexes what it iterates, so this is an
# indexable object rather than a generator: a generator has to be collected at
# every call site, and these sit on the per-sweep path of the whole RHS.
struct LineChunks
    n_lines::Int
    chunk::Int
    n_chunks::Int
end

function _line_chunks(L::Int)
    # No more chunks than lines: a surplus chunk is an empty range that still
    # costs `@threaded` a task spawn.
    n_chunks = clamp(L, 1, Threads.nthreads())
    return LineChunks(L, cld(L, n_chunks), n_chunks)
end

Base.length(chunks::LineChunks) = chunks.n_chunks
Base.eltype(::Type{LineChunks}) = UnitRange{Int}
Base.firstindex(::LineChunks) = 1
Base.lastindex(chunks::LineChunks) = chunks.n_chunks
@inline Base.getindex(chunks::LineChunks, t::Int) =
    (1 + (t - 1) * chunks.chunk):min(t * chunks.chunk, chunks.n_lines)
Base.iterate(chunks::LineChunks, t::Int=1) =
    t > chunks.n_chunks ? nothing : (chunks[t], t + 1)

"""
Row-sweep Thomas over B (lines × n), vectorized across lines. Solves in place and
returns `B`, unchanged for an identity left-hand side. The interface stage is
collective, so every rank of the solver's sub-communicator must call this.
"""
function solve_lines_t!(B::AbstractMatrix{T}, line_solver::LineSolver{T}) where {T}
    line_solver.explicit && return B   # identity LHS: the fill is already the answer
    L, n = size(B)
    F = line_solver.F
    @threaded n*L for rng in _line_chunks(L)
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
    _reduced_solve!(line_solver, L)
    zp, zn = line_solver.zbp, line_solver.zbn
    v, w = line_solver.v, line_solver.w
    @threaded n*L for rng in _line_chunks(L)
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

"""
Row-sweep banded elimination over B (lines × n), vectorized across lines. Solves
in place and returns `B`. The interface stage is collective, so every rank of the
solver's sub-communicator must call this.
"""
function solve_lines_t!(B::AbstractMatrix{T}, line_solver::BandLineSolver{T}) where {T}
    L, n = size(B)
    q = line_solver.q
    F = line_solver.F
    @threaded n*L for rng in _line_chunks(L)
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
    _reduced_solve!(line_solver, L)
    V, W = line_solver.V, line_solver.W
    @threaded n*L for rng in _line_chunks(L)
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
