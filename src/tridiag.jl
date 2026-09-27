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
#   Evaluating at the first and last local rows yields, per line, a
#   2P × 2P "reduced" system in the interface unknowns
#   z = (x₁⁽⁰⁾, xₙ⁽⁰⁾, x₁⁽¹⁾, xₙ⁽¹⁾, …), block-tridiagonal in P (block-cyclic
#   on a periodic line). The reduced matrix depends only on the scheme, so it
#   is assembled from one Allgather of (v₁, vₙ, w₁, wₙ) and factorized once
#   at plan time, as a pivoted band LU (ReducedBand below). Per application:
#   batched local Thomas solves (threaded over lines), a single Allgather of
#   two interface values per line, one band solve for all lines at once, and a
#   threaded rank-local correction. Periodic single-rank lines reuse the same
#   path with self-coupling (no communication) and a dense 2 × 2 LU;
#   non-periodic single-rank lines skip the reduced stage entirely (v = w = 0).

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

# The reduced interface matrix, of order 2qP, couples the 2q interface unknowns
# of a rank (its first and last q values) to the tail of the previous rank and
# the head of the next, and to nothing else. It is therefore block-tridiagonal
# in P with 2q × 2q blocks on a closed line, and block-cyclic-tridiagonal on a
# periodic one. Every rank holds the whole matrix after the plan-time Allgather
# and solves it for each of its own lines, so a dense LU costs 8q²P² operations
# per line; a band LU with partial pivoting costs O(q²P).
#
# The periodic corner blocks are brought into the band by ordering the blocks
# 0, P-1, 1, P-2, 2, …, which places every pair of ring neighbours at most two
# blocks apart, and by reversing the unknowns inside the blocks of the
# descending half of the ring (P-1, P-2, …), whose neighbours lie the other
# way round. The half-bandwidth grows from 2 to 4 for q = 1 and from 5 to 9
# for q = 2, and one pivoted band factorization serves both topologies, where
# bordering or a Sherman–Morrison–Woodbury correction for the corner blocks
# would add a second solve path whose stability rests on the non-cyclic
# part alone.
#
# P = 1 keeps the dense LU. Its matrix is one full 2q × 2q block, so the band
# form saves nothing, and keeping it leaves every serial periodic solve
# bitwise as it was.
struct ReducedBand{T}
    n::Int                 # 2qP
    kl::Int                # lower half-bandwidth in the solve order
    ku::Int                # upper half-bandwidth before pivoting
    ku_used::Int           # U's upper half-bandwidth after it, at most kl + ku
    AB::Matrix{T}        # (2kl+ku+1) × n: A[i, j] at AB[kl+ku+1+i-j, j]; after
                           # factorization U above row kl+ku+2 and L below, as
                           # LAPACK's gbtrf leaves them
    ipiv::Vector{Int}      # row interchanged with row j at elimination step j
    dinv::Vector{T}        # reciprocals of U's diagonal
    pos::Vector{Int}       # solve-order position of each natural-order unknown
    prev_pos::Vector{Int}  # positions of the previous rank's tail (q)
    next_pos::Vector{Int}  # positions of the next rank's head (q)
    stop::Int              # back substitution ends at the first needed position
    x::Matrix{T}           # lines × n: the reduced unknowns, solve order
end

# Rank blocks in solve order. The closed line keeps the natural order; the
# periodic one interleaves the two ends of the ring, as the note above says.
function _reduced_block_order(P::Int, periodic::Bool)
    periodic || return collect(0:(P-1))
    order = Int[]
    lo, hi = 0, P - 1
    while lo <= hi
        push!(order, lo)
        lo += 1
        lo <= hi || break
        push!(order, hi)
        hi -= 1
    end
    return order
end

# Call `add!(row, col, value)` for every entry of the reduced matrix in the
# natural order z = (head₀, tail₀, head₁, tail₁, …), from `allb`, the four q×q
# spike corner blocks of each rank (V_head, V_tail, W_head, W_tail, each
# column-major) as the plan-time Allgather leaves them. An entry may be added
# twice, which happens on periodic lines of one or two ranks.
function _reduced_entries!(add!, allb::Vector{T}, q::Int, P::Int) where {T}
    m2 = 2q
    for rk in 0:(P-1)
        base = 4q * q * rk
        getb(offset, r, t) = allb[base + offset * q * q + (t - 1) * q + r]
        rh = m2 * rk           # head rows offset of rank rk
        rt = m2 * rk + q       # tail rows offset
        for r in 1:q
            add!(rh + r, rh + r, one(T))
            add!(rt + r, rt + r, one(T))
        end
        cprev = m2 * mod(rk - 1, P) + q   # prev tail columns offset
        cnext = m2 * mod(rk + 1, P)       # next head columns offset
        for t in 1:q, r in 1:q
            add!(rh + r, cprev + t, getb(0, r, t))   # V_head
            add!(rt + r, cprev + t, getb(1, r, t))   # V_tail
            add!(rh + r, cnext + t, getb(2, r, t))   # W_head
            add!(rt + r, cnext + t, getb(3, r, t))   # W_tail
        end
    end
    return nothing
end

"""
    _reduced_factor(allb, q, P, p, periodic, lines) -> (red, band)

Factorize the reduced interface matrix from the gathered spike corner blocks:
the dense LU `red` when `P == 1`, otherwise the band LU `band` of rank `p`,
with a workspace for `lines` right-hand sides. The other slot is `nothing`.
"""
function _reduced_factor(allb::Vector{T}, q::Int, P::Int, p::Int,
                         periodic::Bool, lines::Int) where {T}
    if P == 1
        R = zeros(T, 2q, 2q)
        _reduced_entries!((i, j, v) -> (R[i, j] += v), allb, q, P)
        return lu!(R), nothing
    end
    return nothing, ReducedBand(allb, q, P, p, periodic, lines)
end

function ReducedBand(allb::Vector{T}, q::Int, P::Int, p::Int, periodic::Bool,
                     lines::Int) where {T}
    m2 = 2q
    n = m2 * P
    pos = zeros(Int, n)
    for (k, rk) in enumerate(_reduced_block_order(P, periodic))
        reversed = periodic && iseven(k)   # the descending half of the ring
        for r in 1:m2
            pos[m2*rk+r] = m2 * (k - 1) + (reversed ? m2 + 1 - r : r)
        end
    end
    # The half-bandwidths are those of the entries present: a closed line's
    # end ranks gather zero spikes toward the missing neighbour, and a zero
    # entry is left out rather than widening the band to the corner.
    rows, cols, vals = Int[], Int[], T[]
    _reduced_entries!(allb, q, P) do i, j, v
        iszero(v) && return nothing
        push!(rows, pos[i]); push!(cols, pos[j]); push!(vals, v)
        return nothing
    end
    kl = maximum(rows .- cols)
    ku = maximum(cols .- rows)
    kv = kl + ku
    AB = zeros(T, 2kl + ku + 1, n)
    for (i, j, v) in zip(rows, cols, vals)
        AB[kv+1+i-j, j] += v
    end
    ipiv = zeros(Int, n)
    _band_lu!(AB, ipiv, n, kl, ku)
    dinv = T[inv(AB[kv+1, j]) for j in 1:n]
    # Pivoting widens U's band to kl + ku only where rows are interchanged,
    # and a diagonally dominant reduced matrix interchanges none. The back
    # substitution runs over the band U occupies; the entries beyond it are
    # exact zeros, so leaving them out changes no result.
    ku_used = 0
    for j in 1:n, i in max(1, j - kv):(j-1)
        iszero(AB[kv+1+i-j, j]) || (ku_used = max(ku_used, j - i))
    end
    cprev = m2 * mod(p - 1, P) + q
    cnext = m2 * mod(p + 1, P)
    prev_pos = [pos[cprev+t] for t in 1:q]
    next_pos = [pos[cnext+t] for t in 1:q]
    stop = min(minimum(prev_pos), minimum(next_pos))
    ReducedBand{T}(n, kl, ku, ku_used, AB, ipiv, dinv, pos, prev_pos, next_pos,
                   stop, zeros(T, lines, n))
end

# Band LU with partial pivoting in place, the unblocked algorithm of LAPACK's
# gbtf2: row j is swapped with the row of largest magnitude among j..j+kl,
# the interchanges are not applied to earlier multipliers, and U's upper
# half-bandwidth grows to kl + ku, which the kl rows above the band hold.
function _band_lu!(AB::Matrix{T}, ipiv::Vector{Int}, n::Int, kl::Int, ku::Int) where {T}
    kv = kl + ku
    ju = 1   # last column the eliminations so far have reached
    @inbounds for j in 1:n
        km = min(kl, n - j)
        jp = 0
        amax = abs(AB[kv+1, j])
        for i in 1:km
            a = abs(AB[kv+1+i, j])
            if a > amax
                jp, amax = i, a
            end
        end
        ipiv[j] = j + jp
        iszero(AB[kv+1+jp, j]) && error("singular reduced interface matrix")
        ju = max(ju, min(j + ku + jp, n))
        if jp != 0
            for c in j:ju
                a = AB[kv+1+j+jp-c, c]
                AB[kv+1+j+jp-c, c] = AB[kv+1+j-c, c]
                AB[kv+1+j-c, c] = a
            end
        end
        if km > 0
            r = inv(AB[kv+1, j])
            for i in 1:km
                AB[kv+1+i, j] *= r
            end
            for c in (j+1):ju
                u = AB[kv+1+j-c, c]
                for i in 1:km
                    AB[kv+1+j+i-c, c] -= AB[kv+1+i, j] * u
                end
            end
        end
    end
    return AB
end

# Lines per block of the band solve. One elimination step touches a
# contiguous run of this many values in each of kl + ku + 1 columns, which
# stays in L1. Timed at 4096 lines, P = 4 to 64, q = 1 and 2: 16 lines was
# about 2x slower than 128 and 64 lines 10-20% slower, and 256 was within the
# noise of 128 while leaving fewer blocks to thread over.
const REDUCED_BLOCK = 128

# The factorized band solve for lines lo:hi of `x` (lines × n, solve order),
# the forward sweep over every row and the back substitution over the band U
# occupies and down to `band.stop` only, since no row above the first needed
# unknown is read. Each
# line's arithmetic is the same whatever block it falls in, so the result is
# bitwise independent of the number of lines per call, which the batched
# device plan relies on (see `_reduced_ldiv!`).
function _band_solve_block!(x::Matrix{T}, band::ReducedBand{T}, lo::Int, hi::Int) where {T}
    n, kl, kv = band.n, band.kl, band.kl + band.ku
    AB, ipiv, dinv, s = band.AB, band.ipiv, band.dinv, band.stop
    @inbounds for j in 1:(n-1)
        ip = ipiv[j]
        if ip != j
            @simd for l in lo:hi
                a = x[l, j]
                x[l, j] = x[l, ip]
                x[l, ip] = a
            end
        end
        for i in 1:min(kl, n - j)
            m = AB[kv+1+i, j]
            @simd for l in lo:hi
                x[l, j+i] -= m * x[l, j]
            end
        end
    end
    @inbounds for j in n:-1:s
        d = dinv[j]
        @simd for l in lo:hi
            x[l, j] *= d
        end
        for i in max(s, j - band.ku_used):(j-1)
            u = AB[kv+1+i-j, j]
            @simd for l in lo:hi
                x[l, i] -= u * x[l, j]
            end
        end
    end
    return x
end

# Pack the gathered interface values of lines 1:L into the solve order, solve,
# and leave the previous rank's tail and the next rank's head of each line in
# `zbp` and `zbn` (lines × q, or a vector for q = 1). Threaded over blocks of
# lines, which share nothing.
function _band_reduced!(band::ReducedBand{T}, gath::Array{T,3}, zbp, zbn,
                        q::Int, L::Int) where {T}
    m2, P = size(gath, 1), size(gath, 3)
    x, pos = band.x, band.pos
    @threaded L*band.n for b in 1:cld(L, REDUCED_BLOCK)
        lo = (b - 1) * REDUCED_BLOCK + 1
        hi = min(lo + REDUCED_BLOCK - 1, L)
        @inbounds for rk in 0:(P-1), r in 1:m2
            k = pos[m2*rk+r]
            for l in lo:hi
                x[l, k] = gath[r, l, rk+1]
            end
        end
        _band_solve_block!(x, band, lo, hi)
        @inbounds for t in 1:q
            kp, kn = band.prev_pos[t], band.next_pos[t]
            for l in lo:hi
                zbp[l, t] = x[l, kp]
                zbn[l, t] = x[l, kn]
            end
        end
    end
    return nothing
end

# The factorized matrix multiplied back out, dense, in the natural order: the
# eliminations undone from the last to the first, each followed by its row
# interchange. A diagnostic for `bench/reducedsolve.jl`, not a solve path.
function _band_matrix(band::ReducedBand{T}) where {T}
    n, kl, kv = band.n, band.kl, band.kl + band.ku
    X = zeros(T, n, n)
    for j in 1:n, i in max(1, j - kv):j
        X[i, j] = band.AB[kv+1+i-j, j]
    end
    for j in (n-1):-1:1
        for i in 1:min(kl, n - j), c in 1:n
            X[j+i, c] += band.AB[kv+1+i, j] * X[j, c]
        end
        ip = band.ipiv[j]
        if ip != j
            for c in 1:n
                X[j, c], X[ip, c] = X[ip, c], X[j, c]
            end
        end
    end
    return X[band.pos, band.pos]
end

# The reduced matrix of a line solver, dense in the natural order, or
# `nothing` without a reduced stage; for diagnostics.
function _reduced_matrix(line_solver)
    band = line_solver.band
    band === nothing || return _band_matrix(band)
    red = line_solver.red
    return red === nothing ? nothing : Matrix(red)
end

mutable struct LineSolver{T}
    n::Int
    F::TriFactor{T}
    v::Vector{T}          # spike from left coupling aL
    w::Vector{T}          # spike from right coupling cR
    explicit::Bool        # identity LHS: no local solve, no interface stage
    hasred::Bool
    red::Union{RedLU{T}, Nothing}        # dense LU of the reduced matrix (P == 1)
    band::Union{ReducedBand{T}, Nothing} # its band LU (P > 1)
    comm::MPI.Comm        # sub-communicator along the dimension
    P::Int
    p::Int                # 0-based rank within sub-communicator
    lines::Int
    ends::Matrix{T}       # 2 × lines: local (y₁, yₙ) per line
    gath::Array{T,3}      # 2 × lines × P: gathered interface values
    z::Matrix{T}          # 2 × lines: reduced RHS / solution of the dense path
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
    red = band = nothing
    if hasred
        # The spike corners are the q = 1 corner blocks (V_head, V_tail,
        # W_head, W_tail) of `_reduced_entries!`.
        quad = T[v[1], v[n], w[1], w[n]]
        allq = zeros(T, 4P)
        if P > 1
            MPI.Allgather!(quad, MPI.UBuffer(allq, 4), comm)
        else
            allq .= quad
        end
        red, band = _reduced_factor(allq, 1, P, p, periodic, lines)
    end
    LineSolver{T}(n, F, v, w, explicit, hasred, red, band, comm, P, p, lines,
                  zeros(T, 2, lines), zeros(T, 2, lines, max(P, 1)),
                  zeros(T, red === nothing ? 0 : 2, lines),
                  zeros(T, lines), zeros(T, lines))
end

# The factorized dense solve of the reduced system for `L` right-hand-side
# columns, in blocks of `Lb` columns. A batched device plan gathers the ends
# of every tile's lines into one `z` (lines_device.jl), and LAPACK's
# triangular solves are not bitwise invariant to the number of right-hand
# sides they are handed, so each tile's block is solved with the column
# count the tile's own host plan would pass; `Lb == L` is the one call every
# unbatched plan has always made. The band solve needs no blocking.
function _reduced_ldiv!(red, z::AbstractMatrix, L::Int, Lb::Int)
    Lb == L && return ldiv!(red, z)
    for c0 in 1:Lb:L
        ldiv!(red, view(z, :, c0:(c0 + Lb - 1)))
    end
    return z
end

_interface_width(::LineSolver) = 1

# The Allgather of the reduced stage: every rank's `ends` (2q × L) into `gath`.
function _gather_ends!(line_solver, L::Int)
    q = _interface_width(line_solver)
    if line_solver.P > 1
        MPI.Allgather!(line_solver.ends,
                       MPI.UBuffer(vec(line_solver.gath), 2q * L), line_solver.comm)
    else
        copyto!(view(line_solver.gath, :, :, 1), line_solver.ends)
    end
    return nothing
end

# The rest of the reduced stage, from the gathered interface values to the
# correction values in `zbp` and `zbn`: the band solve when P > 1, the dense
# LU otherwise. Local; it reads `gath` and does not modify it.
function _solve_reduced!(line_solver, L::Int, Lb::Int)
    q = _interface_width(line_solver)
    # `band` and `red` are small unions (see RedLU above); every caller sits
    # behind `hasred`, which the constructor pairs with one of the two
    # factorizations, so the narrowing checks are dead at runtime. They close
    # the unions at the solve calls, which JET otherwise reports at every
    # solver entry point.
    band = line_solver.band
    if band !== nothing
        _band_reduced!(band, line_solver.gath, line_solver.zbp, line_solver.zbn, q, L)
        return nothing
    end
    red = line_solver.red
    red === nothing && error("reduced solve without a reduced factorization")
    m2 = 2q
    z, gath = line_solver.z, line_solver.gath
    @inbounds for l in 1:L, r in 1:m2
        z[r, l] = gath[r, l, 1]
    end
    _reduced_ldiv!(red, z, L, Lb)
    # One rank: the previous rank's tail is this rank's tail, the next rank's
    # head its head.
    @inbounds for l in 1:L, t in 1:q
        line_solver.zbp[l, t] = z[q+t, l]
        line_solver.zbn[l, t] = z[t, l]
    end
    return nothing
end

"""
Reduced interface stage shared by every layout of the line solve: from
`line_solver.ends` holding the first and last q local values of each line,
gather the interface values, solve the reduced system, and leave the
correction values each line needs, the previous rank's last q unknowns and the
next rank's first q, in `line_solver.zbp` and `line_solver.zbn`. The gather is
collective when `P > 1`, so every rank of the sub-communicator must call this.
`Lb` is the column block of the dense `P == 1` solve (see `_reduced_ldiv!`).
"""
function _reduced_solve!(line_solver::LineSolver, L::Int, Lb::Int=L)
    _gather_ends!(line_solver, L)
    _solve_reduced!(line_solver, L, Lb)
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
