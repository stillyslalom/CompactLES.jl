# 3-D MPI Cartesian domain decomposition.
#
# Global grid of n_global points per dimension is split evenly across a process
# grid `dims`. Each rank stores its interior block padded by n_halo halo layers
# on every side. Sub-communicators along each dimension support the distributed
# tridiagonal solves and line-wise collectives.

struct Decomp
    comm::MPI.Comm                      # Cartesian communicator
    dims::NTuple{3,Int}                 # process grid
    coords::NTuple{3,Int}               # 0-based Cartesian coordinates of this rank
    periodic::NTuple{3,Bool}
    n_global::NTuple{3,Int}
    n_local::NTuple{3,Int}              # local interior extents
    offset::NTuple{3,Int}               # global index = offset[d] + local index
    n_halo::Int                         # halo width (active dimensions)
    active::NTuple{3,Bool}              # n_global[d] > 1; collapsed dims carry no
                                        # derivatives, halos, or exchanges
    n_halo_d::NTuple{3,Int}             # per-dim halo pad: n_halo if active else 0
    neighbors::NTuple{3,NTuple{2,Int}}  # (lo, hi) ranks per dim (PROC_NULL at open edges)
    sub::NTuple{3,MPI.Comm}             # sub-communicator along each dim
    sub_rank::NTuple{3,Int}
    sub_size::NTuple{3,Int}
    send_buf::Vector{Vector{Float64}}   # per-dim halo send buffers
    recv_buf::Vector{Vector{Float64}}   # per-dim halo recv buffers
end

"Even split of N points over P ranks; rank r (0-based) gets (count, offset)."
function local_range(N::Int, P::Int, r::Int)
    base, rem = divrem(N, P)
    n = base + (r < rem ? 1 : 0)
    offset = r * base + min(r, rem)
    return n, offset
end

function Decomp(n_global::NTuple{3,Int}, periodic::NTuple{3,Bool};
                dims=nothing, n_halo::Int=4)
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    np = MPI.Comm_size(MPI.COMM_WORLD)
    active = ntuple(d -> n_global[d] > 1, 3)
    nact = count(active)
    if dims === nothing
        # MPI.jl's second argument is the dims ARRAY (zeros = "choose for me"),
        # not the number of dimensions. Passing the count instead silently asks
        # for a 0-dimensional grid: it returns a 0-d array for nact ≤ 2 (which
        # popfirst! below cannot index) and throws MPIError(779) "cannot
        # partition nodes as requested" for nact == 3 — i.e. the default
        # dims=nothing path failed for every 3-D grid, at every rank count.
        adims = nact == 0 ? Int[] : Int.(MPI.Dims_create(np, zeros(Cint, nact)))
        pdims = ntuple(3) do dd
            active[dd] || return 1
            popfirst!(adims)
        end
    else
        pdims = Tuple(collect(Int, dims))
        all(d -> active[d] || pdims[d] == 1, 1:3) ||
            error("collapsed dimensions must not be decomposed: dims=$pdims")
    end
    prod(pdims) == np ||
        error("process grid $pdims does not match communicator size $np")
    comm = MPI.Cart_create(MPI.COMM_WORLD, collect(Int, pdims);
                           periodic=collect(periodic), reorder=true)
    coords = Tuple(Int.(MPI.Cart_coords(comm)))
    neighbors = ntuple(d -> Tuple(Int.(MPI.Cart_shift(comm, d - 1, 1))), 3)  # (src=lo, dst=hi)
    nl_off = ntuple(d -> local_range(n_global[d], pdims[d], coords[d]), 3)
    n_local = ntuple(d -> nl_off[d][1], 3)
    offset  = ntuple(d -> nl_off[d][2], 3)
    sub = ntuple(d -> MPI.Cart_sub(comm, [k == d for k in 1:3]), 3)
    sub_rank = ntuple(d -> MPI.Comm_rank(sub[d]), 3)
    sub_size = ntuple(d -> MPI.Comm_size(sub[d]), 3)
    n_halo_d = ntuple(d -> active[d] ? n_halo : 0, 3)
    full = ntuple(d -> n_local[d] + 2*n_halo_d[d], 3)
    cnt(d) = active[d] ? n_halo * prod(full[k] for k in 1:3 if k != d) : 0
    send_buf = [zeros(cnt(d)) for d in 1:3]
    recv_buf = [zeros(cnt(d)) for d in 1:3]
    Decomp(comm, pdims, coords, periodic, n_global, n_local, offset, n_halo, active, n_halo_d,
           neighbors, sub, sub_rank, sub_size, send_buf, recv_buf)
end

"CartesianIndices of the interior (halo-offset) block."
interior(decomp::Decomp) = CartesianIndices(
    ntuple(d -> decomp.n_halo_d[d]+1:decomp.n_halo_d[d]+decomp.n_local[d], 3))

"Allocate a scalar field with halos (collapsed dims carry no padding)."
field(decomp::Decomp) = zeros(ntuple(d -> decomp.n_local[d] + 2*decomp.n_halo_d[d], 3))

"Allocate a conserved state Q(x,y,z,1:n_cons) with halos."
allocate_state(decomp::Decomp, n_cons::Int) =
    zeros(ntuple(d -> decomp.n_local[d] + 2*decomp.n_halo_d[d], 3)..., n_cons)

"Is this rank at the closed (non-periodic) global low/high edge of dim d?"
at_lo_edge(decomp::Decomp, d::Int) =
    decomp.active[d] && !decomp.periodic[d] && decomp.sub_rank[d] == 0
at_hi_edge(decomp::Decomp, d::Int) =
    decomp.active[d] && !decomp.periodic[d] && decomp.sub_rank[d] == decomp.sub_size[d] - 1
