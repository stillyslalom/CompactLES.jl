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

# Columns interleaved per solve_cols! below: the recurrence in each column is
# a dependent multiply-add chain, so a lone column exposes the full FMA
# latency at every row while the arithmetic units sit idle. Sweeping a small
# block of independent columns row by row fills the pipeline (measured 1.5-1.7x
# on the x-sweep solve of a 64^3 RHS; widths 16 and 32 tie, 8 is slightly
# behind). Per column the operations and their order
# match solve_col! exactly, so the result is bitwise identical to it; only
# the interleaving across columns differs, and the device `colwise` mirror
# of the x-sweep arithmetic is unaffected.
const COL_BLOCK = 16

"In-place Thomas solve of columns `lo:hi` of `B` (n × lines), interleaved."
function solve_cols!(B::AbstractMatrix{T}, F::TriFactor{T},
                     lo::Int, hi::Int) where {T}
    n = F.n
    l, dinv, c = F.l, F.dinv, F.c
    @inbounds for i in 2:n
        li = l[i]
        for col in lo:hi
            B[i, col] -= li * B[i-1, col]
        end
    end
    @inbounds begin
        dn = dinv[n]
        for col in lo:hi
            B[n, col] *= dn
        end
    end
    @inbounds for i in (n-1):-1:1
        ci = c[i]
        di = dinv[i]
        for col in lo:hi
            B[i, col] = (B[i, col] - ci * B[i+1, col]) * di
        end
    end
    return B
end

# Concrete type of `lu!` on a dense Matrix{T}. Typing `red` as a small union of
# this and Nothing (rather than Any) keeps the solver structs concrete without
# a type parameter that would ripple into every plan struct that holds one.
const RedLU{T} = LU{T, Matrix{T}, Vector{Int}}

mutable struct LineSolver{T}
    n::Int
    F::TriFactor{T}
    v::Vector{T}          # spike from left coupling aL
    w::Vector{T}          # spike from right coupling cR
    explicit::Bool        # identity LHS: no local solve, no interface stage
    hasred::Bool
    red::Union{RedLU{T}, Nothing}   # LU of the reduced matrix, or nothing
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
    LineSolver(a, b, c, aL, cR, comm, P, p, lines; periodic, explicit=false)

`a`, `b`, `c` are the local sub/diag/super-diagonals, each of length n, with the
closure rows substituted where this rank owns a closed global edge and
the ghost coupling folded onto the diagonal where it owns a parity fold.
`aL` is the coupling of local row 1 to the previous rank's last unknown, `cR`
that of local row n to the next rank's first unknown; `aL` is zero where this
rank owns a closed low edge or a low fold, and `cR` likewise at the high edge.

`comm` is the sub-communicator along the dimension, `P` its size, and `p` this
rank's 0-based position in it. `lines` is the number of right-hand sides that
[`solve_lines!`](@ref) will carry, and sizes the interface workspaces allocated
here. `periodic` marks the dimension as wrapping, which retains the reduced
interface stage even at `P == 1`. Unless `explicit` is set, assembling that
stage is collective when `P > 1`, so every rank of `comm` must construct the
solver.

`explicit` asserts that the left-hand side is the identity, allowing both the
local solve and the interface stage to be skipped. It must be derived from the
scheme alone and never from this rank's edge status: the interface stage
contains a collective, and a flag that some ranks set and others do not
causes a deadlock. `aL` and `cR` are rank-dependent quantities of
this kind, so they are not consulted here.
"""
function LineSolver(a::Vector{T}, b::Vector{T}, c::Vector{T},
                    aL::T, cR::T, comm::MPI.Comm, P::Int, p::Int,
                    lines::Int; periodic::Bool, explicit::Bool=false) where {T}
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
    hasred = !explicit && ((P > 1) || periodic)
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
    LineSolver{T}(n, F, v, w, explicit, hasred, red, comm, P, p, lines,
                  zeros(T, 2, lines), zeros(T, 2, lines, max(P, 1)),
                  zeros(T, 2P, lines), zeros(T, lines), zeros(T, lines))
end

"""
Reduced interface stage shared by every layout of the tridiagonal solve: from
`line_solver.ends` holding the local `(y₁, yₙ)` of each line, gather the
interface values, solve the dense reduced system, and leave the two correction
values each line needs, the previous rank's last unknown and the next rank's
first, in `line_solver.zbp` and `line_solver.zbn`. The gather is collective
when `P > 1`, so every rank of the sub-communicator must call this.
"""
function _reduced_solve!(line_solver::LineSolver{T}, L::Int) where {T}
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
    # `red` is a small union (see RedLU above); every caller sits behind
    # `hasred`, which the constructor pairs with the factorization, so the
    # narrowing check is dead at runtime. It closes the union at the
    # `ldiv!` call, which JET otherwise reports at every solver entry point.
    red = line_solver.red
    red === nothing && error("reduced solve without a reduced factorization")
    ldiv!(red, line_solver.z)
    cprev = 2 * mod(line_solver.p - 1, line_solver.P) + 2
    cnext = 2 * mod(line_solver.p + 1, line_solver.P) + 1
    @inbounds for l in 1:L
        line_solver.zbp[l] = line_solver.z[cprev, l]
        line_solver.zbn[l] = line_solver.z[cnext, l]
    end
    return nothing
end

"""
    solve_lines!(B, line_solver)

Solve the (possibly distributed) tridiagonal system for every column of
`B` (n × lines) in place, and return `B`. MPI collectives run from the serial
section only, and every rank of the solver's sub-communicator must call this.

An identity left-hand side (`line_solver.explicit`) returns `B` unchanged, since
the caller's fill is the solution.
"""
function solve_lines!(B::AbstractMatrix{T}, line_solver::LineSolver{T}) where {T}
    line_solver.explicit && return B   # identity LHS: the fill is the answer
    n, L = size(B)
    @threaded n*L for b in 1:cld(L, COL_BLOCK)
        lo = (b - 1) * COL_BLOCK + 1
        solve_cols!(B, line_solver.F, lo, min(lo + COL_BLOCK - 1, L))
    end
    line_solver.hasred || return B

    @inbounds for l in 1:L
        line_solver.ends[1, l] = B[1, l]
        line_solver.ends[2, l] = B[n, l]
    end
    _reduced_solve!(line_solver, L)
    v, w = line_solver.v, line_solver.w
    zp, zn = line_solver.zbp, line_solver.zbn
    @threaded n*L for l in 1:L
        xl = zp[l]
        xr = zn[l]
        @inbounds for i in 1:n
            B[i, l] -= v[i] * xl + w[i] * xr
        end
    end
    return B
end
