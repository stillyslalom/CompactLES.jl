# The implicit diffusion stage of reference/IMPLICIT.md on one patch: the
# solve of (I − γΔt L) T = r, with L = Σ_d J⁻¹ G C K D_s the staggered
# operator of staggered.jl summed over the active dimensions. None of it is
# reached from the right-hand side.
#
# Weighted form. With V the node volumes (J times the node weights: h per
# dimension, h/2 on a wall node), V L is symmetric negative semidefinite on
# periodic lines, walls, symmetry planes, the spherical origin, stretched lines
# and curvilinear shells (staggered.jl), so V (I − γΔt L) is symmetric
# positive definite there and the stage is solved as V A T = V r by
# preconditioned conjugate gradients in the Euclidean inner product. At a fold
# whose area vanishes oddly (the cylindrical axis, the spherical poles) V L is
# symmetric only up to a defect in the rows next to the fold, and the same
# weighted system is solved by restarted GMRES, right-preconditioned by the
# same operator. The choice is made from the folds alone, so every rank makes
# it identically, and a symmetric closure at the axis would move those
# geometries onto the conjugate-gradient path without other change.
#
# Preconditioner. S = V (I − γΔt L₂), with L₂ the second-order conservative
# three-point operator on the same metric: the face factor C of the staggered
# operator times the arithmetic mean of κ over the two nodes of a face. S is a
# symmetric M-matrix, (S x)_I = m_I x_I + Σ_faces g (x_I − x_J), with m the
# node volume and g = γΔt C κ̄ times the face area over the spacing. A fold
# carries no face: at a coordinate singularity C vanishes on the plane, and
# across a symmetry plane the even temperature has no jump. At a wall of odd
# parity (T − T_wall) the wall node is a Dirichlet node: its faces are dropped,
# the interior neighbour keeps g on its diagonal, and its row is m x = b.
#
# S is inverted approximately by one multigrid V-cycle. The coarse spaces
# are aggregates of two nodes per coarsened dimension, formed within a rank's
# block, with piecewise-constant prolongation P and restriction Pᵀ. The coarse
# operator keeps the form above: the aggregate masses sum, and the
# conductances of the fine faces between two aggregates sum into one coarse
# face, as in the Galerkin product Pᵀ S P, and are then halved along a
# coarsened dimension. The Galerkin conductance is twice the one a
# discretization on the coarse spacing gives, since a piecewise-constant
# function carries its whole jump across one face; the Galerkin correction of
# a smooth error falls short by that factor, and the shortfall compounds over
# the levels of a V-cycle. The halved operator is the coarse discretization,
# and it bounds the correction by twice the S-orthogonal projection. Every
# level is therefore a seven-point operator whose lines are tridiagonal.
#
# The smoother is zebra line relaxation along each dimension in turn: the
# grid lines of one colour (the parity of their transverse indices) solved
# exactly with their neighbours' current values, then the other colour, the
# lines crossing ranks through the spike method of tridiag.jl on per-line
# coefficients (`VariableLines`). Line relaxation is exact along a dominant
# direction; the colouring damps the error that alternates between lines,
# which relaxing every line at once leaves. The descent relaxes the
# dimensions and colours in one order and the ascent in the reverse, and the
# coarsest level is solved by a dense Cholesky factorization replicated on
# every rank, so the cycle is a fixed symmetric positive definite operator and
# conjugate gradients need no flexible variant. The angular dimensions of a
# curvilinear metric are coarsened last (`_stage_levels`).

const STAGE_RESTART = 20      # GMRES basis length before a restart

@inline _unit3(k::Int) = CartesianIndex(ntuple(i -> i == k ? 1 : 0, 3))
@inline _transverse(d::Int) = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
@inline _line_point(d::Int, i::Int, a::Int, b::Int) =
    d == 1 ? CartesianIndex(i, a, b) : d == 2 ? CartesianIndex(a, i, b) :
    CartesianIndex(a, b, i)

# ---------------------------------------------------------------------------
# Tridiagonal lines with per-line coefficients
# ---------------------------------------------------------------------------

"""
    VariableLines

Distributed tridiagonal solve of `lines` systems of local extent `n`, each
with its own coefficients, along one dimension. Row `i` of line `l` reads
`lower[i, l] x[i-1] + diag[i, l] x[i] + upper[i, l] x[i+1]`; `lower[1, l]`
couples to the previous rank's last unknown (or the wrapped one) and
`upper[n, l]` to the next rank's first, zero at a closed end. The local
blocks are factorized by the Thomas algorithm and the ranks coupled by the
spike method of `LineSolver`, whose reduced system here differs per line and
is factorized per line (dense, of order `2P`, without pivoting: the lines of
a diagonally dominant M-matrix give spikes of modulus below one, and the
reduced matrix is diagonally dominant by rows).
"""
struct VariableLines{T}
    n::Int
    lines::Int
    lower::Matrix{T}
    diag::Matrix{T}
    upper::Matrix{T}
    mult::Matrix{T}           # Thomas multipliers
    dinv::Matrix{T}           # inverses of the eliminated diagonal
    v::Matrix{T}              # spike of the coupling to the previous rank
    w::Matrix{T}              # spike of the coupling to the next rank
    hasred::Bool
    reduced::Array{T,3}       # (2P, 2P, lines) LU factors
    quad::Matrix{T}           # (4, lines): v₁, vₙ, w₁, wₙ
    allq::Array{T,3}          # (4, lines, P)
    ends::Matrix{T}           # (2, lines)
    gath::Array{T,3}          # (2, lines, P)
    z::Matrix{T}              # (2P, lines)
    prev_last::Vector{T}
    next_first::Vector{T}
    comm::MPI.Comm
    P::Int
    p::Int
    periodic::Bool
end

function VariableLines{T}(n::Int, lines::Int, comm::MPI.Comm, P::Int, p::Int,
                          periodic::Bool) where {T}
    m() = zeros(T, n, lines)
    hasred = P > 1 || periodic
    VariableLines{T}(n, lines, m(), m(), m(), m(), m(), m(), m(), hasred,
                     zeros(T, hasred ? 2P : 0, hasred ? 2P : 0, lines),
                     zeros(T, 4, lines), zeros(T, 4, lines, P), zeros(T, 2, lines),
                     zeros(T, 2, lines, P), zeros(T, 2P, lines), zeros(T, lines),
                     zeros(T, lines), comm, P, p, periodic)
end

@inline function _thomas_column!(X::Matrix{T}, vl::VariableLines{T}, l::Int) where {T}
    n = vl.n
    @inbounds begin
        for i in 2:n
            X[i, l] -= vl.mult[i, l] * X[i-1, l]
        end
        X[n, l] *= vl.dinv[n, l]
        for i in (n-1):-1:1
            X[i, l] = (X[i, l] - vl.upper[i, l] * X[i+1, l]) * vl.dinv[i, l]
        end
    end
    return X
end

_ring_prev(q, P, periodic) = q > 0 ? q - 1 : (periodic ? P - 1 : -1)
_ring_next(q, P, periodic) = q < P - 1 ? q + 1 : (periodic ? 0 : -1)

# Factorize the coefficients in `lower`, `diag`, `upper`. Collective over the
# sub-communicator when the reduced stage is present and the set holds lines;
# every rank of the sub-communicator holds the same number, so an empty set
# (one colour of a single line) skips the collective everywhere.
function factor_lines!(vl::VariableLines{T}) where {T}
    vl.lines == 0 && return vl
    n = vl.n
    @threaded n*vl.lines for l in 1:vl.lines
        @inbounds begin
            vl.dinv[1, l] = inv(vl.diag[1, l])
            for i in 2:n
                μ = vl.lower[i, l] * vl.dinv[i-1, l]
                vl.mult[i, l] = μ
                vl.dinv[i, l] = inv(vl.diag[i, l] - μ * vl.upper[i-1, l])
            end
            for i in 1:n
                vl.v[i, l] = zero(T); vl.w[i, l] = zero(T)
            end
            vl.v[1, l] = vl.lower[1, l]
            vl.w[n, l] = vl.upper[n, l]
        end
        _thomas_column!(vl.v, vl, l)
        _thomas_column!(vl.w, vl, l)
    end
    vl.hasred || return vl
    @inbounds for l in 1:vl.lines
        vl.quad[1, l] = vl.v[1, l]; vl.quad[2, l] = vl.v[n, l]
        vl.quad[3, l] = vl.w[1, l]; vl.quad[4, l] = vl.w[n, l]
    end
    if vl.P > 1
        MPI.Allgather!(vl.quad, MPI.UBuffer(vl.allq, 4 * vl.lines), vl.comm)
    else
        copyto!(vl.allq, vl.quad)
    end
    P = vl.P; m = 2P
    @threaded m*m*vl.lines for l in 1:vl.lines
        R = view(vl.reduced, :, :, l)
        @inbounds begin
            fill!(R, zero(T))
            for q in 0:P-1
                i1 = 2q + 1; iN = 2q + 2
                R[i1, i1] += one(T); R[iN, iN] += one(T)
                prev = _ring_prev(q, P, vl.periodic)
                next = _ring_next(q, P, vl.periodic)
                if prev >= 0
                    R[i1, 2prev+2] += vl.allq[1, l, q+1]
                    R[iN, 2prev+2] += vl.allq[2, l, q+1]
                end
                if next >= 0
                    R[i1, 2next+1] += vl.allq[3, l, q+1]
                    R[iN, 2next+1] += vl.allq[4, l, q+1]
                end
            end
            for k in 1:m-1
                piv = R[k, k]
                for i in k+1:m
                    f = R[i, k] / piv
                    R[i, k] = f
                    f == 0 && continue
                    for j in k+1:m
                        R[i, j] -= f * R[k, j]
                    end
                end
            end
        end
    end
    return vl
end

# Solve every column of `B` (n × lines) in place. Collective over the
# sub-communicator when the reduced stage is present.
function solve_variable_lines!(B::Matrix{T}, vl::VariableLines{T}) where {T}
    vl.lines == 0 && return B
    n = vl.n
    @threaded n*vl.lines for l in 1:vl.lines
        _thomas_column!(B, vl, l)
    end
    vl.hasred || return B
    @inbounds for l in 1:vl.lines
        vl.ends[1, l] = B[1, l]; vl.ends[2, l] = B[n, l]
    end
    if vl.P > 1
        MPI.Allgather!(vl.ends, MPI.UBuffer(vl.gath, 2 * vl.lines), vl.comm)
    else
        copyto!(vl.gath, vl.ends)
    end
    P = vl.P; m = 2P
    prev = _ring_prev(vl.p, P, vl.periodic)
    next = _ring_next(vl.p, P, vl.periodic)
    @threaded m*m*vl.lines for l in 1:vl.lines
        @inbounds begin
            for q in 0:P-1
                vl.z[2q+1, l] = vl.gath[1, l, q+1]
                vl.z[2q+2, l] = vl.gath[2, l, q+1]
            end
            for i in 2:m, j in 1:i-1
                vl.z[i, l] -= vl.reduced[i, j, l] * vl.z[j, l]
            end
            for i in m:-1:1
                s = vl.z[i, l]
                for j in i+1:m
                    s -= vl.reduced[i, j, l] * vl.z[j, l]
                end
                vl.z[i, l] = s / vl.reduced[i, i, l]
            end
            xl = prev >= 0 ? vl.z[2prev+2, l] : zero(T)
            xr = next >= 0 ? vl.z[2next+1, l] : zero(T)
            for i in 1:n
                B[i, l] -= vl.v[i, l] * xl + vl.w[i, l] * xr
            end
        end
    end
    return B
end

# ---------------------------------------------------------------------------
# Multigrid levels of S
# ---------------------------------------------------------------------------

# One level: local extent `n`, a halo of one node on every active dimension,
# the operator (`mass`, and `cond[d]` the conductance of the face between
# node I and I + e_d, the lower neighbour's face in halo slot 0), the cycle's
# vectors, and the line solvers of the relaxation, one per active dimension
# (none on the coarsest level).
struct StageLevel{T}
    n::NTuple{3,Int}
    pad::NTuple{3,Int}
    coarsen::NTuple{3,Bool}
    dims::Vector{Int}
    mass::Array{T,3}
    cond::NTuple{3,Array{T,3}}
    x::Array{T,3}
    b::Array{T,3}
    r::Array{T,3}
    members::Vector{NTuple{2,Vector{NTuple{2,Int}}}}   # per dimension and colour
    lines::Vector{NTuple{2,VariableLines{T}}}
    B::Vector{NTuple{2,Matrix{T}}}
    send::Vector{T}
    recv::Vector{T}
end

# The lines along each active dimension, coloured by the parity of the sum
# of their two transverse indices in the level's global numbering, so that a
# half-sweep over one colour updates lines whose transverse neighbours all
# carry the other (zebra relaxation); an odd periodic extent leaves one
# neighbouring pair of a colour at the wrap. Collective.
function StageLevel{T}(n, coarsen, decomp::Decomp, relax::Bool) where {T}
    dims = [d for d in 1:3 if decomp.active[d]]
    pad = ntuple(d -> decomp.active[d] ? 1 : 0, 3)
    ext = n .+ 2 .* pad
    f() = zeros(T, ext)
    offset = ntuple(3) do k
        decomp.active[k] || return 0
        counts = MPI.Allgather(n[k], decomp.sub[k])
        sum(counts[1:decomp.sub_rank[k]]; init=0)
    end
    members = NTuple{2,Vector{NTuple{2,Int}}}[]
    lines = NTuple{2,VariableLines{T}}[]
    B = NTuple{2,Matrix{T}}[]
    if relax
        for d in dims
            t1, t2 = _transverse(d)
            colours = (NTuple{2,Int}[], NTuple{2,Int}[])
            for b in 1:n[t2], a in 1:n[t1]
                push!(colours[1 + (a + offset[t1] + b + offset[t2]) % 2], (a, b))
            end
            push!(members, colours)
            push!(lines, ntuple(c -> VariableLines{T}(n[d], length(colours[c]),
                                                      decomp.sub[d], decomp.sub_size[d],
                                                      decomp.sub_rank[d],
                                                      decomp.periodic[d]), 2))
            push!(B, ntuple(c -> zeros(T, n[d], length(colours[c])), 2))
        end
    end
    slab = maximum((prod(ext) ÷ ext[d] for d in dims); init=0)
    StageLevel{T}(n, pad, coarsen, dims, f(), (f(), f(), f()), f(), f(), f(),
                  members, lines, B, zeros(T, slab), zeros(T, slab))
end

_level_interior(lev::StageLevel) =
    CartesianIndices(ntuple(k -> lev.pad[k]+1:lev.pad[k]+lev.n[k], 3))

# The halos of a level array along `d`, from the neighbours or the periodic
# wrap; a closed end's halo is left as it is.
function _level_exchange!(f::Array{T,3}, lev::StageLevel{T}, decomp::Decomp,
                          d::Int) where {T}
    decomp.active[d] || return f
    n = lev.n[d]
    if decomp.sub_size[d] == 1
        decomp.periodic[d] || return f
        selectdim(f, d, 1) .= selectdim(f, d, n + 1)
        selectdim(f, d, n + 2) .= selectdim(f, d, 2)
        return f
    end
    lo, hi = decomp.neighbors[d]
    len = length(selectdim(f, d, 1))
    sv = view(lev.send, 1:len); rv = view(lev.recv, 1:len)
    hi != PNULL && copyto!(sv, selectdim(f, d, n + 1))
    MPI.Sendrecv!(sv, rv, decomp.comm; dest=hi, source=lo,
                  sendtag=300 + 2d, recvtag=300 + 2d)
    lo != PNULL && copyto!(selectdim(f, d, 1), rv)
    lo != PNULL && copyto!(sv, selectdim(f, d, 2))
    MPI.Sendrecv!(sv, rv, decomp.comm; dest=lo, source=hi,
                  sendtag=301 + 2d, recvtag=301 + 2d)
    hi != PNULL && copyto!(selectdim(f, d, n + 2), rv)
    return f
end

# The line coefficients of the relaxation along each dimension, and their
# factorization (collective along the dimension).
function _factor_level!(lev::StageLevel{T}) where {T}
    pad = CartesianIndex(lev.pad)
    for (di, d) in enumerate(lev.dims), colour in 1:2
        vl = lev.lines[di][colour]
        ed = _unit3(d)
        @inbounds for (l, (a, b)) in enumerate(lev.members[di][colour])
            for i in 1:lev.n[d]
                I = _line_point(d, i, a, b) + pad
                diag = lev.mass[I]
                for k in lev.dims
                    ek = _unit3(k)
                    diag += lev.cond[k][I] + lev.cond[k][I-ek]
                end
                vl.diag[i, l] = diag
                vl.lower[i, l] = -lev.cond[d][I-ed]
                vl.upper[i, l] = -lev.cond[d][I]
            end
        end
        factor_lines!(vl)
    end
    return lev
end

# One half-sweep of the relaxation along the `di`-th active dimension: every
# line of one colour solved exactly with its transverse neighbours at their
# current values.
function _relax!(lev::StageLevel{T}, di::Int, colour::Int, decomp::Decomp) where {T}
    d = lev.dims[di]
    for k in lev.dims
        k == d || _level_exchange!(lev.x, lev, decomp, k)
    end
    B = lev.B[di][colour]; vl = lev.lines[di][colour]
    members = lev.members[di][colour]
    pad = CartesianIndex(lev.pad)
    x, b = lev.x, lev.b
    @threaded vl.n*vl.lines for l in 1:vl.lines
        a, c = members[l]
        @inbounds for i in 1:vl.n
            I = _line_point(d, i, a, c) + pad
            acc = b[I]
            for k in lev.dims
                k == d && continue
                ek = _unit3(k)
                acc += lev.cond[k][I] * x[I+ek] + lev.cond[k][I-ek] * x[I-ek]
            end
            B[i, l] = acc
        end
    end
    solve_variable_lines!(B, vl)
    @threaded vl.n*vl.lines for l in 1:vl.lines
        a, c = members[l]
        @inbounds for i in 1:vl.n
            x[_line_point(d, i, a, c) + pad] = B[i, l]
        end
    end
    return lev
end

# r = b − S x.
function _level_residual!(lev::StageLevel{T}, decomp::Decomp) where {T}
    for k in lev.dims
        _level_exchange!(lev.x, lev, decomp, k)
    end
    x = lev.x
    @inbounds for I in _level_interior(lev)
        s = lev.mass[I] * x[I]
        for k in lev.dims
            ek = _unit3(k)
            s += lev.cond[k][I] * (x[I] - x[I+ek]) + lev.cond[k][I-ek] * (x[I] - x[I-ek])
        end
        lev.r[I] = lev.b[I] - s
    end
    return lev
end

@inline _aggregate(lev::StageLevel, I::CartesianIndex{3}) =
    CartesianIndex(ntuple(k -> lev.coarsen[k] ? (I[k] + 1) >> 1 : I[k], 3))

# The operator of the next level: aggregate masses, and the conductances of
# the faces leaving an aggregate upward along each dimension, halved along a
# coarsened one (header of this file).
function _coarsen_operator!(coarse::StageLevel{T}, fine::StageLevel{T},
                            decomp::Decomp) where {T}
    half = T(1) / 2
    fill!(coarse.mass, zero(T))
    foreach(c -> fill!(c, zero(T)), coarse.cond)
    pf = CartesianIndex(fine.pad); pc = CartesianIndex(coarse.pad)
    @inbounds for I in CartesianIndices(fine.n)
        Ic = _aggregate(fine, I) + pc
        coarse.mass[Ic] += fine.mass[I+pf]
        for k in fine.dims
            last = !fine.coarsen[k] || I[k] == 2 * ((I[k] + 1) >> 1) || I[k] == fine.n[k]
            last && (coarse.cond[k][Ic] += (fine.coarsen[k] ? half : one(T)) *
                                           fine.cond[k][I+pf])
        end
    end
    for k in coarse.dims
        _level_exchange!(coarse.cond[k], coarse, decomp, k)
    end
    return coarse
end

function _restrict!(coarse::StageLevel{T}, fine::StageLevel{T}) where {T}
    fill!(coarse.b, zero(T))
    pf = CartesianIndex(fine.pad); pc = CartesianIndex(coarse.pad)
    @inbounds for I in CartesianIndices(fine.n)
        coarse.b[_aggregate(fine, I) + pc] += fine.r[I+pf]
    end
    return coarse
end

function _prolong!(fine::StageLevel{T}, coarse::StageLevel{T}) where {T}
    pf = CartesianIndex(fine.pad); pc = CartesianIndex(coarse.pad)
    @inbounds for I in CartesianIndices(fine.n)
        fine.x[I+pf] += coarse.x[_aggregate(fine, I) + pc]
    end
    return fine
end

# The coarsest level, replicated: every rank assembles the whole operator
# from one reduction of the local masses and conductances and factorizes it,
# and each solve reduces the right-hand side and back-substitutes.
mutable struct CoarseSolve{T}
    offset::Int
    npoints::Int
    total::Int
    neighbor::Matrix{Int}          # (3, total): global index across each upper face
    values::Vector{T}              # (4 total): mass and the three conductances
    matrix::Matrix{T}
    factor::Cholesky{T,Matrix{T}}
    rhs::Vector{T}
end

function CoarseSolve{T}(lev::StageLevel{T}, decomp::Decomp) where {T}
    npoints = prod(lev.n)
    counts = MPI.Allgather(npoints, decomp.comm)
    rank = MPI.Comm_rank(decomp.comm)
    offset = sum(counts[1:rank]; init=0)
    total = sum(counts)
    # Global indices through the level's halo exchange; zero across a closed end.
    index = lev.x
    fill!(index, zero(T))
    pad = CartesianIndex(lev.pad)
    for (m, I) in enumerate(CartesianIndices(lev.n))
        index[I+pad] = T(offset + m)
    end
    for k in lev.dims
        _level_exchange!(index, lev, decomp, k)
    end
    neighbor = zeros(Int, 3, total)
    for (m, I) in enumerate(CartesianIndices(lev.n))
        for k in lev.dims
            neighbor[k, offset+m] = round(Int, index[I+pad+_unit3(k)])
        end
    end
    MPI.Allreduce!(neighbor, +, decomp.comm)
    fill!(index, zero(T))
    matrix = Matrix{T}(LinearAlgebra.I, total, total)
    CoarseSolve{T}(offset, npoints, total, neighbor, zeros(T, 4total), matrix,
                   cholesky!(Symmetric(copy(matrix))), zeros(T, total))
end

function _assemble_coarse!(cs::CoarseSolve{T}, lev::StageLevel{T},
                           decomp::Decomp) where {T}
    v = cs.values
    fill!(v, zero(T))
    pad = CartesianIndex(lev.pad)
    for (m, I) in enumerate(CartesianIndices(lev.n))
        g = cs.offset + m
        v[4g-3] = lev.mass[I+pad]
        for k in lev.dims
            v[4g-3+k] = lev.cond[k][I+pad]
        end
    end
    MPI.Allreduce!(v, +, decomp.comm)
    M = cs.matrix
    fill!(M, zero(T))
    @inbounds for g in 1:cs.total
        M[g, g] += v[4g-3]
        for k in 1:3
            q = cs.neighbor[k, g]
            q == 0 && continue
            c = v[4g-3+k]
            M[g, g] += c; M[q, q] += c
            M[g, q] -= c; M[q, g] -= c
        end
    end
    cs.factor = cholesky!(Symmetric(M))
    return cs
end

function _coarse_solve!(lev::StageLevel{T}, cs::CoarseSolve{T}, decomp::Decomp) where {T}
    rhs = cs.rhs
    fill!(rhs, zero(T))
    pad = CartesianIndex(lev.pad)
    for (m, I) in enumerate(CartesianIndices(lev.n))
        rhs[cs.offset+m] = lev.b[I+pad]
    end
    MPI.Allreduce!(rhs, +, decomp.comm)
    ldiv!(cs.factor, rhs)
    for (m, I) in enumerate(CartesianIndices(lev.n))
        lev.x[I+pad] = rhs[cs.offset+m]
    end
    return lev
end

# The hierarchy of a decomposition: a dimension is coarsened while every
# rank holds at least two nodes along it, so the coarsest level holds one
# node per rank along every active dimension, or as few as a block admits.
# A dimension in `last` is coarsened only once no other can be: the angular
# dimensions of a curvilinear metric, whose scale factor carries the radius.
# Away from the axis or origin their couplings are the weak ones and the two
# other directions dominate. Line relaxation does not smooth an error that
# varies slowly along both dominant directions and oscillates along the weak
# one, and a coarse level that keeps the weak dimension represents it
# (bench/implicitstage.jl). Near the spherical origin the two angular
# directions dominate the radial one, which neither order of coarsening
# covers.
function _stage_levels(decomp::Decomp{T}, last::NTuple{3,Bool}) where {T}
    levels = StageLevel{T}[]
    n = decomp.n_local
    while true
        least = ntuple(d -> decomp.active[d] ?
                            MPI.Allreduce(n[d], min, decomp.comm) : 1, 3)
        coarsen = ntuple(d -> least[d] >= 2, 3)
        any(coarsen .& .!last) && (coarsen = coarsen .& .!last)
        push!(levels, StageLevel{T}(n, coarsen, decomp, any(coarsen)))
        any(coarsen) || break
        n = ntuple(d -> coarsen[d] ? cld(n[d], 2) : n[d], 3)
    end
    return levels
end

# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------

"""
    DiffusionStage(solver; parity=1)

The implicit diffusion stage `(I − γΔt L) T_ion = r` of a single-patch
`solver` on host storage, with `L` the staggered conduction operator
`J⁻¹ ∂(C κ ∂T)` of every active dimension (`StaggeredDiffusion`), and the
workspace of its solve: the weighted Krylov vectors, the multigrid hierarchy
of the second-order preconditioner, and a GMRES basis at a fold whose area
vanishes oddly (the cylindrical axis, the spherical poles), where the
operator is not symmetric. `parity` is the temperature's parity at the walls
of every closed dimension: +1 the adiabatic wall, -1 the isothermal wall, for
which the temperature is written as `T_ion − T_wall` and vanishes on the wall
nodes. Solved by `solve_stage!`; `stage_operator!` applies the operator.
"""
struct DiffusionStage{T}
    decomp::Decomp{T}
    dims::Vector{Int}
    ops::Vector{Any}                    # a StaggeredDiffusion per active dimension
    parity::Int
    symmetric::Bool
    h::NTuple{3,T}
    volume::Array{T,3}                  # V at the nodes, zero in the halos
    inverse_volume::Array{T,3}          # 1/V at the nodes, zero in the halos
    node_weight::NTuple{3,Vector{T}}    # h, and h/2 on a wall node
    wall::NTuple{3,NTuple{2,Bool}}      # this rank owns a wall end (lo, hi)
    dirichlet::Array{Bool,3}            # the wall nodes of an odd temperature
    applied::Array{T,3}
    weighted_rhs::Array{T,3}
    residual::Array{T,3}
    precond::Array{T,3}
    direction::Array{T,3}
    product::Array{T,3}
    basis::Vector{Array{T,3}}           # GMRES only
    hessenberg::Matrix{T}
    givens::Matrix{T}                   # (2, restart): cosines and sines
    projected::Vector{T}
    dots::Vector{T}
    levels::Vector{StageLevel{T}}
    coarse::CoarseSolve{T}
    gamma_dt::Base.RefValue{T}
    capacity::Array{T,3}                # m at the nodes when `weighted[]`
    weighted::Base.RefValue{Bool}       # whether the stage carries a capacity
end

function DiffusionStage(solver; parity::Int=1)
    abs(parity) == 1 || throw(ArgumentError("parity must be ±1, not $parity"))
    length(solver.patches) == 1 || throw(ArgumentError(
        "the implicit stage runs on a single-patch solver"))
    _cpu_storage(solver.inv_J) || throw(ArgumentError(
        "the implicit stage runs on host storage only"))
    decomp = solver.decomp
    T = eltype(solver.inv_J)
    dims = [d for d in 1:3 if decomp.active[d]]
    # Held untyped: the solve then compiles once per element type rather than
    # once per geometry, and the one dispatch per operator application is
    # negligible against its line solves.
    ops = Any[StaggeredDiffusion(solver, d; parity=parity) for d in dims]
    symmetric = _stage_symmetric(solver, dims)
    wall = ntuple(3) do k
        closed = decomp.active[k] && !decomp.periodic[k]
        fold = solver.folds[k]
        (closed && at_lo_edge(decomp, k) && (fold === nothing || !fold.lo),
         closed && at_hi_edge(decomp, k) && (fold === nothing || !fold.hi))
    end
    node_weight = ntuple(3) do k
        w = fill(T(solver.h[k]), decomp.n_local[k])
        wall[k][1] && (w[1] /= 2)
        wall[k][2] && (w[end] /= 2)
        w
    end
    volume = field(decomp)
    pad = decomp.n_halo_d
    for I in CartesianIndices(decomp.n_local)
        J = I + CartesianIndex(pad)
        volume[J] = node_weight[1][I[1]] * node_weight[2][I[2]] * node_weight[3][I[3]] /
                    solver.inv_J[J]
    end
    dirichlet = falses(decomp.n_local...)
    if parity == -1
        for I in CartesianIndices(decomp.n_local), k in dims
            (wall[k][1] && I[k] == 1 || wall[k][2] && I[k] == decomp.n_local[k]) &&
                (dirichlet[I] = true)
        end
    end
    levels = _stage_levels(decomp, ntuple(k -> !unit_scalefactor(solver.metric, k), 3))
    coarse = CoarseSolve{T}(levels[end], decomp)
    restart = symmetric ? 0 : STAGE_RESTART
    inverse_volume = field(decomp)
    inverse_volume[interior(decomp)] .= inv.(volume[interior(decomp)])
    DiffusionStage(decomp, dims, ops, parity, symmetric, T.(solver.h), volume,
                   inverse_volume, node_weight, wall,
                   Array(dirichlet), field(decomp), field(decomp), field(decomp),
                   field(decomp), field(decomp), field(decomp),
                   [field(decomp) for _ in 1:(symmetric ? 0 : restart + 1)],
                   zeros(T, restart + 1, restart), zeros(T, 2, restart),
                   zeros(T, restart + 1), zeros(T, max(restart + 1, 2)),
                   levels, coarse, Ref(zero(T)), field(decomp), Ref(false))
end

# The capacity of the next assembly and operator application: `nothing` for
# the unit capacity of `(I − γΔt L)`, or a node field `m` for `(m − γΔt L)`.
function _set_capacity!(stage::DiffusionStage, capacity)
    stage.weighted[] = capacity !== nothing
    capacity === nothing && return stage
    for I in interior(stage.decomp)
        stage.capacity[I] = capacity[I]
    end
    return stage
end

# Whether V L is symmetric along every active dimension: everywhere but at a
# fold across which the flux C K D_s T is continued evenly while D_s T is
# odd, the cylindrical axis and the spherical poles (staggered.jl). A
# symmetric closure there moves those geometries onto conjugate gradients
# through this test alone.
_stage_symmetric(solver, dims) =
    all(d -> solver.folds[d] === nothing ||
             solver.folds[d].sigflux[solver.equations.i_energy] == -1, dims)

# The fine level of S from the coefficient and γΔt, the coarse levels by
# aggregation, and every factorization. Collective.
function _assemble_stage!(stage::DiffusionStage{T}, kappa, gamma_dt) where {T}
    decomp = stage.decomp
    stage.gamma_dt[] = T(gamma_dt)
    for d in stage.dims
        exchange_dim!(kappa, decomp, d)
    end
    fine = stage.levels[1]
    fill!(fine.mass, zero(T))
    foreach(c -> fill!(c, zero(T)), fine.cond)
    pad = CartesianIndex(decomp.n_halo_d)
    pl = CartesianIndex(fine.pad)
    n = decomp.n_local
    for (op, d) in zip(stage.ops, stage.dims)
        _fine_conductance!(fine.cond[d], stage, op.face, kappa, d, T(gamma_dt))
    end
    if stage.weighted[]
        @inbounds for I in CartesianIndices(n)
            fine.mass[I+pl] = stage.volume[I+pad] * stage.capacity[I+pad]
        end
    else
        @inbounds for I in CartesianIndices(n)
            fine.mass[I+pl] = stage.volume[I+pad]
        end
    end
    if stage.parity == -1
        # Drop the faces of a Dirichlet node; an interior neighbour keeps the
        # conductance on its diagonal.
        @inbounds for I in CartesianIndices(n)
            stage.dirichlet[I] || continue
            for k in stage.dims
                ek = _unit3(k)
                # The face above; a neighbour off the block along k lies on
                # the same wall, a Dirichlet node on another rank.
                J = I + ek
                J[k] <= n[k] && !stage.dirichlet[J] &&
                    (fine.mass[J+pl] += fine.cond[k][I+pl])
                fine.cond[k][I+pl] = zero(T)
                # The face below, where this rank holds it.
                I[k] >= 2 || continue
                J = I - ek
                stage.dirichlet[J] || (fine.mass[J+pl] += fine.cond[k][J+pl])
                fine.cond[k][J+pl] = zero(T)
            end
        end
    end
    for k in stage.dims
        _level_exchange!(fine.cond[k], fine, decomp, k)
    end
    levels = stage.levels
    for ℓ in 1:length(levels)-1
        _factor_level!(levels[ℓ])
        _coarsen_operator!(levels[ℓ+1], levels[ℓ], decomp)
    end
    _assemble_coarse!(stage.coarse, levels[end], decomp)
    return stage
end

# γΔt C κ̄ times the face area over the spacing, on the faces above each node
# along `d` that lie inside the domain; `face` is C at the midpoints, or
# nothing on a unit line.
function _fine_conductance!(cond, stage::DiffusionStage{T}, face, kappa, d::Int,
                            gamma_dt::T) where {T}
    decomp = stage.decomp
    pad = CartesianIndex(decomp.n_halo_d)
    pl = CartesianIndex(stage.levels[1].pad)
    n = decomp.n_local
    nw = stage.node_weight
    ed = _unit3(d)
    closed_hi = at_hi_edge(decomp, d)
    t1, t2 = _transverse(d)
    scale = gamma_dt / stage.h[d]
    @inbounds for I in CartesianIndices(n)
        (closed_hi && I[d] == n[d]) && continue
        J = I + pad
        C = face === nothing ? one(T) : face[J]
        area = nw[t1][I[t1]] * nw[t2][I[t2]]
        cond[I+pl] = scale * C * area * (kappa[J] + kappa[J+ed]) / 2
    end
    return cond
end

# x = C b on level ℓ: one V-cycle from a zero start, recursively.
function _cycle!(stage::DiffusionStage{T}, ℓ::Int) where {T}
    decomp = stage.decomp
    levels = stage.levels
    lev = levels[ℓ]
    if ℓ == length(levels)
        return _coarse_solve!(lev, stage.coarse, decomp)
    end
    fill!(lev.x, zero(T))
    for di in eachindex(lev.dims), colour in 1:2
        _relax!(lev, di, colour, decomp)
    end
    _level_residual!(lev, decomp)
    coarse = levels[ℓ+1]
    _restrict!(coarse, lev)
    _cycle!(stage, ℓ + 1)
    _prolong!(lev, coarse)
    for di in reverse(eachindex(lev.dims)), colour in 2:-1:1
        _relax!(lev, di, colour, decomp)
    end
    return lev
end

# z = B v with B the multigrid cycle on S, over the interior. Collective.
function _precondition!(z, stage::DiffusionStage{T}, v) where {T}
    fine = stage.levels[1]
    pad = CartesianIndex(stage.decomp.n_halo_d)
    pl = CartesianIndex(fine.pad)
    @inbounds for I in CartesianIndices(stage.decomp.n_local)
        fine.b[I+pl] = v[I+pad]
    end
    _cycle!(stage, 1)
    @inbounds for I in CartesianIndices(stage.decomp.n_local)
        z[I+pad] = fine.x[I+pl]
    end
    return z
end

# out = V (m x − γΔt Σ_d L_d x), with m = 1 unless the stage carries a
# capacity. Collective.
function _apply_weighted!(out, stage::DiffusionStage{T}, x, kappa) where {T}
    γ = stage.gamma_dt[]
    if stage.weighted[]
        out .= stage.capacity .* x
    else
        out .= x
    end
    for op in stage.ops
        staggered_diffusion!(stage.applied, op, x, kappa, stage.decomp)
        out .-= γ .* stage.applied
    end
    out .*= stage.volume
    return out
end

function _local_dot(u, v, decomp::Decomp{T}) where {T}
    s = zero(T)
    @inbounds for I in interior(decomp)
        s += u[I] * v[I]
    end
    return s
end

_global_dot(u, v, decomp::Decomp) = MPI.Allreduce(_local_dot(u, v, decomp), +, decomp.comm)

# The inner product of two weighted vectors (V times a node field) that is the
# volume integral of the product of the node fields: the L2 norm of the
# residual in which both methods stop, and the inner product of GMRES.
function _local_dot(u, v, stage::DiffusionStage{T}) where {T}
    s = zero(T)
    w = stage.inverse_volume
    @inbounds for I in interior(stage.decomp)
        s += u[I] * v[I] * w[I]
    end
    return s
end

_global_dot(u, v, stage::DiffusionStage) =
    MPI.Allreduce(_local_dot(u, v, stage), +, stage.decomp.comm)

"""
    diffusion_operator!(out, stage, T_ion, kappa)

Write `L T_ion` into the interior of `out`, with `L` the staggered
conduction operator of `stage` (a `DiffusionStage`) summed over the active
dimensions on the coefficient `kappa`. Exchanges the halos of `T_ion` and
`kappa`, so every rank must call it.
"""
function diffusion_operator!(out, stage::DiffusionStage{T}, T_ion, kappa) where {T}
    pad = CartesianIndex(stage.decomp.n_halo_d)
    for (m, op) in enumerate(stage.ops)
        staggered_diffusion!(stage.applied, op, T_ion, kappa, stage.decomp)
        @inbounds for I in CartesianIndices(stage.decomp.n_local)
            J = I + pad
            out[J] = m == 1 ? stage.applied[J] : out[J] + stage.applied[J]
        end
    end
    return out
end

"""
    stage_operator!(out, stage, T_ion, kappa, gamma_dt; capacity=nothing)

Write `m T_ion − γΔt L T_ion` into the interior of `out`, with `L` the
staggered conduction operator of `stage` (a `DiffusionStage`) on the
coefficient `kappa`, `gamma_dt` the implicit weight times the step, and `m`
the node field `capacity`, or one when it is `nothing`. Exchanges the halos
of `T_ion` and `kappa`, so every rank must call it.
"""
function stage_operator!(out, stage::DiffusionStage{T}, T_ion, kappa, gamma_dt;
                         capacity=nothing) where {T}
    stage.gamma_dt[] = T(gamma_dt)
    _set_capacity!(stage, capacity)
    _apply_weighted!(out, stage, T_ion, kappa)
    pad = CartesianIndex(stage.decomp.n_halo_d)
    @inbounds for I in CartesianIndices(stage.decomp.n_local)
        out[I+pad] /= stage.volume[I+pad]
    end
    return out
end

"""
    solve_stage!(T_ion, stage, rhs, kappa, gamma_dt; rtol=1e-10, maxiter=200,
                 capacity=nothing)

Solve `(m − γΔt L) T_ion = rhs` for the interior of `T_ion`, starting from
its current values, with `L` the staggered conduction operator of `stage` (a
`DiffusionStage`) on the coefficient `kappa`, `gamma_dt` the implicit weight
times the step, and `m` the positive node field `capacity`, or one when it
is `nothing`. The volume weighting then carries `m`, so the system stays
symmetric wherever the unit-capacity one is. The iteration stops when the L2
norm of the residual over the domain (the volume integral of its square)
falls below `rtol` times that of `rhs`. Returns
`(converged, iterations, residual)`, the last relative to `rhs`; every rank
returns the same values, and every rank must call it. Where the stage's walls
carry an odd temperature, `rhs` is `m (T_ion − T_wall)` and vanishes on the
wall nodes.
"""
function solve_stage!(T_ion, stage::DiffusionStage{T}, rhs, kappa, gamma_dt;
                      rtol::Real=1e-10, maxiter::Int=200, capacity=nothing) where {T}
    _set_capacity!(stage, capacity)
    _assemble_stage!(stage, kappa, gamma_dt)
    decomp = stage.decomp
    b = stage.weighted_rhs
    b .= stage.volume .* rhs
    if stage.parity == -1
        pad = CartesianIndex(decomp.n_halo_d)
        @inbounds for I in CartesianIndices(decomp.n_local)
            J = I + pad
            stage.dirichlet[I] || continue
            T_ion[J] = stage.weighted[] ? rhs[J] / stage.capacity[J] : rhs[J]
        end
    end
    bnorm = sqrt(_global_dot(b, b, stage))
    if bnorm == 0
        T_ion .= zero(T)
        return (converged=true, iterations=0, residual=zero(T))
    end
    stage.symmetric ? _pcg!(T_ion, stage, b, kappa, T(rtol) * bnorm, maxiter, bnorm) :
                      _gmres!(T_ion, stage, b, kappa, T(rtol) * bnorm, maxiter, bnorm)
end

function _pcg!(x, stage::DiffusionStage{T}, b, kappa, tol, maxiter, bnorm) where {T}
    decomp = stage.decomp
    r, z, p, q = stage.residual, stage.precond, stage.direction, stage.product
    _apply_weighted!(q, stage, x, kappa)
    r .= b .- q
    rnorm = sqrt(_global_dot(r, r, stage))
    rnorm <= tol && return (converged=true, iterations=0, residual=rnorm / bnorm)
    _precondition!(z, stage, r)
    p .= z
    ρ = _global_dot(r, z, decomp)
    for it in 1:maxiter
        _apply_weighted!(q, stage, p, kappa)
        α = ρ / _global_dot(p, q, decomp)
        x .+= α .* p
        r .-= α .* q
        _precondition!(z, stage, r)
        stage.dots[1] = _local_dot(r, z, decomp)
        stage.dots[2] = _local_dot(r, r, stage)
        MPI.Allreduce!(view(stage.dots, 1:2), +, decomp.comm)
        ρnew, rr = stage.dots[1], stage.dots[2]
        rnorm = sqrt(rr)
        rnorm <= tol && return (converged=true, iterations=it, residual=rnorm / bnorm)
        p .= z .+ (ρnew / ρ) .* p
        ρ = ρnew
    end
    return (converged=false, iterations=maxiter, residual=rnorm / bnorm)
end

# Restarted GMRES in the volume-weighted inner product, right-preconditioned,
# with classical Gram–Schmidt applied twice so that each pass is one batched
# reduction.
function _gmres!(x, stage::DiffusionStage{T}, b, kappa, tol, maxiter, bnorm) where {T}
    decomp = stage.decomp
    V = stage.basis; H = stage.hessenberg; G = stage.givens; g = stage.projected
    m = size(H, 2)
    r, z, w = stage.residual, stage.precond, stage.product
    total = 0
    rnorm = zero(T)
    while true
        _apply_weighted!(w, stage, x, kappa)
        r .= b .- w
        rnorm = sqrt(_global_dot(r, r, stage))
        (rnorm <= tol || total >= maxiter) && break
        V[1] .= r ./ rnorm
        fill!(g, zero(T)); g[1] = rnorm
        fill!(H, zero(T))
        j = 0
        while j < m && total < maxiter
            j += 1; total += 1
            _precondition!(z, stage, V[j])
            _apply_weighted!(V[j+1], stage, z, kappa)
            for pass in 1:2
                dots = view(stage.dots, 1:j)
                for i in 1:j
                    dots[i] = _local_dot(V[i], V[j+1], stage)
                end
                MPI.Allreduce!(dots, +, decomp.comm)
                for i in 1:j
                    H[i, j] += dots[i]
                    V[j+1] .-= dots[i] .* V[i]
                end
            end
            hn = sqrt(_global_dot(V[j+1], V[j+1], stage))
            H[j+1, j] = hn
            hn > 0 && (V[j+1] ./= hn)
            for i in 1:j-1
                c, s = G[1, i], G[2, i]
                H[i, j], H[i+1, j] = c * H[i, j] + s * H[i+1, j], -s * H[i, j] + c * H[i+1, j]
            end
            ρ = hypot(H[j, j], H[j+1, j])
            c, s = H[j, j] / ρ, H[j+1, j] / ρ
            G[1, j] = c; G[2, j] = s
            H[j, j] = ρ; H[j+1, j] = zero(T)
            g[j+1] = -s * g[j]; g[j] = c * g[j]
            (abs(g[j+1]) <= tol || hn == 0) && break
        end
        for i in j:-1:1
            s = g[i]
            for k in i+1:j
                s -= H[i, k] * g[k]
            end
            g[i] = s / H[i, i]
        end
        r .= zero(T)
        for i in 1:j
            r .+= g[i] .* V[i]
        end
        _precondition!(z, stage, r)
        x .+= z
    end
    return (converged=rnorm <= tol, iterations=total, residual=rnorm / bnorm)
end
