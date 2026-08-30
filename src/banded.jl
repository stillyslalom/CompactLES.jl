# Block-banded distributed line solver for compact schemes whose LHS has
# half-bandwidth q ≥ 1 (q = 1 tridiagonal, q = 2 pentadiagonal, ...). This
# generalizes the spike / reduced-interface method of tridiag.jl: cross-rank
# coupling now involves the last q unknowns of the previous rank and the first
# q of the next, through triangular q×q coupling blocks AL and CR built from
# the interior LHS coefficients:
#
#   T x = d − AL·x_prev_tail ⊗ (rows 1..q) − CR·x_next_head ⊗ (rows n−q+1..n)
#
# With Y = T⁻¹d and spike blocks V = T⁻¹[AL padded], W = T⁻¹[CR padded]
# (each n × q), the local solution is
#
#   x = y − V·x_prev_tail − W·x_next_head,
#
# and evaluating at each rank's head (1..q) and tail (n−q+1..n) rows yields a
# dense (2q·P) × (2q·P) reduced system per line in the interface unknowns
# z = (head₀, tail₀, head₁, tail₁, …). As in the tridiagonal case the reduced
# matrix depends only on the scheme: it is assembled from one Allgather of the
# four q×q spike corner blocks per rank and LU-factorized at plan time. Per
# application: batched banded local solves threaded over lines, one Allgather
# of 2q interface values per line, one dense solve for all lines, threaded
# correction. Periodic single-rank lines self-couple with no communication;
# non-periodic single-rank lines skip the reduced stage (V = W = 0).
#
# The banded LU is unpivoted. Compact-scheme LHS matrices (e.g. Lele C10:
# α = 1/2, β = 1/20) are not strictly diagonally dominant but are
# well-conditioned in practice; unpivoted solves are standard for them.

"Banded LU (no pivoting) of a matrix with lower and upper half-bandwidth q,
stored band-wise: Ab[q+1+s, i] = A[i, i+s] for s ∈ −q..q."
struct BandFactor{T}
    n::Int
    q::Int
    L::Matrix{T}      # q × n multipliers; L[m, k] eliminates row k+m via pivot row k
    U::Matrix{T}      # (q+1) × n; U[1+s, i] = U[i, i+s], s ∈ 0..q
end

function BandFactor(Ab::Matrix{T}, q::Int) where {T}
    size(Ab, 1) == 2q + 1 || error("band storage must have 2q+1 rows")
    n = size(Ab, 2)
    Wk = copy(Ab)
    L = zeros(T, q, n)
    @inbounds for k in 1:(n-1)
        piv = Wk[q+1, k]
        mmax = min(q, n - k)
        for m in 1:mmax
            i = k + m
            l = Wk[q+1-m, i] / piv
            L[m, k] = l
            for t in 1:mmax
                # A[i, k+t] -= l * A[k, k+t]; both stay inside the band
                Wk[q+1+t-m, i] -= l * Wk[q+1+t, k]
            end
        end
    end
    U = zeros(T, q + 1, n)
    @inbounds for i in 1:n, s in 0:min(q, n - i)
        U[1+s, i] = Wk[q+1+s, i]
    end
    BandFactor{T}(n, q, L, U)
end

"In-place banded forward and back substitution of one right-hand side."
function solve_col!(x::AbstractVector{T}, F::BandFactor{T}) where {T}
    n, q = F.n, F.q
    L, U = F.L, F.U
    # Skip a zero multiplier, as the transposed sweep does: identical on
    # finite data, and it keeps the x sweep from spreading a NaN or Inf
    # through 0 * x where the y/z sweeps leave the row untouched.
    @inbounds for k in 1:(n-1)
        xk = x[k]
        for m in 1:min(q, n - k)
            lm = L[m, k]
            iszero(lm) && continue
            x[k+m] -= lm * xk
        end
    end
    # Multiply by the reciprocal pivot on every banded
    # path: the transposed y/z sweep hoists `inv(U[1, i])` out of its line
    # loop, and dividing here made the x sweep a different-rounding operator
    # from the y and z sweeps (8.9e-16 on a q = 2, n = 40 line), which cost an
    # exactly symmetric problem its exact directional symmetry under C10.
    @inbounds for i in n:-1:1
        acc = x[i]
        for t in 1:min(q, n - i)
            acc -= U[1+t, i] * x[i+t]
        end
        x[i] = acc * inv(U[1, i])
    end
    return x
end

# Interleaved counterpart of solve_col!, as the tridiagonal solve_cols! in
# tridiag.jl: a block of independent columns swept row by row hides the
# recurrence latency a lone column exposes. Per column the operations and
# their order match solve_col! exactly, so the result is bitwise identical.
"In-place banded solve of columns `lo:hi` of `B` (n × lines), interleaved."
function solve_cols!(B::AbstractMatrix{T}, F::BandFactor{T},
                     lo::Int, hi::Int) where {T}
    n, q = F.n, F.q
    Lb, U = F.L, F.U
    @inbounds for k in 1:(n-1)
        for m in 1:min(q, n - k)
            lm = Lb[m, k]
            iszero(lm) && continue
            for col in lo:hi
                B[k+m, col] -= lm * B[k, col]
            end
        end
    end
    @inbounds for i in n:-1:1
        u0 = inv(U[1, i])            # reciprocal, as `solve_col!` and the
        for col in lo:hi             # transposed sweep: one rounding per path
            acc = B[i, col]
            for t in 1:min(q, n - i)
                acc -= U[1+t, i] * B[i+t, col]
            end
            B[i, col] = acc * u0
        end
    end
    return B
end

mutable struct BandLineSolver{T}
    n::Int
    q::Int
    F::BandFactor{T}
    V::Matrix{T}          # n × q spikes from the left coupling block AL
    W::Matrix{T}          # n × q spikes from the right coupling block CR
    hasred::Bool
    red::Union{RedLU{T}, Nothing}   # LU of the (2qP)² reduced matrix, or nothing
    comm::MPI.Comm
    P::Int
    p::Int                # 0-based rank within the sub-communicator
    lines::Int
    ends::Matrix{T}       # 2q × lines: local (y_head; y_tail) per line
    gath::Array{T,3}      # 2q × lines × P
    z::Matrix{T}          # 2qP × lines
    zbp::Matrix{T}        # lines × q contiguous interface copies for the
    zbn::Matrix{T}        # transposed (vectorized) correction sweep
end

"""
    BandLineSolver(Ab, AL, CR, comm, P, p, lines; periodic)

`Ab` is the local band matrix, (2q+1) × n in the storage
`Ab[q+1+s, i] = A[i, i+s]`, with closure rows substituted where this
rank owns a closed global edge and the ghost couplings folded back where
it owns a parity fold. `AL`, `CR` are the q×q coupling blocks to the previous
rank's tail and the next rank's head; `AL` is zero where this rank owns a closed
low edge or a low fold, and `CR` likewise at the high edge. `q` is taken from
`size(AL, 1)` and `n` from `size(Ab, 2)`, and `n > 2q` is required.

`comm` is the sub-communicator along the dimension, `P` its size, and `p` this
rank's 0-based position in it. `lines` is the number of right-hand sides that
[`solve_lines!`](@ref) will carry, and sizes the interface workspaces allocated
here. `periodic` marks the dimension as wrapping, which retains the reduced
interface stage even at `P == 1`. Assembling that stage is collective when
`P > 1`, so every rank of `comm` must construct the solver. `Ab`, `AL` and `CR`
are read, not modified.
"""
function BandLineSolver(Ab::Matrix{T}, AL::Matrix{T}, CR::Matrix{T},
                        comm::MPI.Comm, P::Int, p::Int, lines::Int;
                        periodic::Bool) where {T}
    q = size(AL, 1)
    n = size(Ab, 2)
    n > 2q || error("local extent $n must exceed 2q = $(2q)")
    F = BandFactor(Ab, q)
    V = zeros(T, n, q)
    W = zeros(T, n, q)
    if !all(iszero, AL)
        for t in 1:q
            V[1:q, t] .= AL[:, t]
            solve_col!(view(V, :, t), F)
        end
    end
    if !all(iszero, CR)
        for t in 1:q
            W[(n-q+1):n, t] .= CR[:, t]
            solve_col!(view(W, :, t), F)
        end
    end
    hasred = (P > 1) || periodic
    red = nothing
    if hasred
        # Pack the four q×q spike corner blocks (V_head, V_tail, W_head,
        # W_tail), column-major each.
        blk = zeros(T, 4q * q)
        idx = 1
        for Mb in (view(V, 1:q, :), view(V, (n-q+1):n, :),
                   view(W, 1:q, :), view(W, (n-q+1):n, :))
            for t in 1:q, r in 1:q
                blk[idx] = Mb[r, t]
                idx += 1
            end
        end
        allb = zeros(T, 4q * q * P)
        if P > 1
            MPI.Allgather!(blk, MPI.UBuffer(allb, 4q * q), comm)
        else
            allb .= blk
        end
        m2 = 2q
        R = zeros(T, m2 * P, m2 * P)
        for rk in 0:(P-1)
            base = 4q * q * rk
            getb(offset, r, t) = allb[base + offset * q * q + (t - 1) * q + r]
            rh = m2 * rk           # head rows offset of rank rk
            rt = m2 * rk + q       # tail rows offset
            for r in 1:q
                R[rh+r, rh+r] += 1
                R[rt+r, rt+r] += 1
            end
            cprev = m2 * mod(rk - 1, P) + q   # prev tail columns offset
            cnext = m2 * mod(rk + 1, P)       # next head columns offset
            for t in 1:q, r in 1:q
                R[rh+r, cprev+t] += getb(0, r, t)   # V_head
                R[rt+r, cprev+t] += getb(1, r, t)   # V_tail
                R[rh+r, cnext+t] += getb(2, r, t)   # W_head
                R[rt+r, cnext+t] += getb(3, r, t)   # W_tail
            end
        end
        red = lu!(R)
    end
    BandLineSolver{T}(n, q, F, V, W, hasred, red, comm, P, p, lines,
                      zeros(T, 2q, lines), zeros(T, 2q, lines, max(P, 1)),
                      zeros(T, 2q * P, lines),
                      zeros(T, lines, q), zeros(T, lines, q))
end

"""
Reduced interface stage shared by every layout of the banded solve: from
`line_solver.ends` holding the local head and tail values (2q per line),
gather the interface values, solve the dense reduced system, and leave the `q`
correction values each line needs from the previous rank's tail and the next
rank's head in `line_solver.zbp` and `line_solver.zbn` (lines × q). The gather
is collective when `P > 1`, so every rank of the sub-communicator must call
this.
"""
function _reduced_solve!(line_solver::BandLineSolver{T}, L::Int) where {T}
    q = line_solver.q
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
    # `red` is a small union (see RedLU in tridiag.jl); every caller sits
    # behind `hasred`, which the constructor pairs with the factorization,
    # so the narrowing check is dead at runtime. It closes the union at the
    # `ldiv!` call, which JET otherwise reports at every solver entry point.
    red = line_solver.red
    red === nothing && error("reduced solve without a reduced factorization")
    ldiv!(red, line_solver.z)
    cprev = m2 * mod(line_solver.p - 1, line_solver.P) + q
    cnext = m2 * mod(line_solver.p + 1, line_solver.P)
    @inbounds for l in 1:L, t in 1:q
        line_solver.zbp[l, t] = line_solver.z[cprev+t, l]
        line_solver.zbn[l, t] = line_solver.z[cnext+t, l]
    end
    return nothing
end

"""
    solve_lines!(B, line_solver::BandLineSolver)

Solve the (possibly distributed) banded system for every column of `B`
(n × lines) in place, and return `B`. MPI collectives run from the serial
section only, and every rank of the solver's sub-communicator must call this.
"""
function solve_lines!(B::AbstractMatrix{T}, line_solver::BandLineSolver{T}) where {T}
    n, L = size(B)
    q = line_solver.q
    @threaded n*L for b in 1:cld(L, COL_BLOCK)
        lo = (b - 1) * COL_BLOCK + 1
        solve_cols!(B, line_solver.F, lo, min(lo + COL_BLOCK - 1, L))
    end
    line_solver.hasred || return B

    @inbounds for l in 1:L, r in 1:q
        line_solver.ends[r, l] = B[r, l]
        line_solver.ends[q+r, l] = B[n-q+r, l]
    end
    _reduced_solve!(line_solver, L)
    V, W = line_solver.V, line_solver.W
    zbp, zbn = line_solver.zbp, line_solver.zbn
    @threaded n*L for l in 1:L
        @inbounds for i in 1:n
            acc = zero(T)
            for t in 1:q
                acc += V[i, t] * zbp[l, t] + W[i, t] * zbn[l, t]
            end
            B[i, l] -= acc
        end
    end
    return B
end
