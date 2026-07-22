# 3-D MPI Cartesian domain decomposition.
#
# Global grid of nglob points per dimension is split evenly across a process
# grid `dims`. Each rank stores its interior block padded by H halo layers on
# every side. Sub-communicators along each dimension support the distributed
# tridiagonal solves and line-wise collectives.

struct Decomp
    comm::MPI.Comm                    # Cartesian communicator
    dims::NTuple{3,Int}               # process grid
    coords::NTuple{3,Int}             # 0-based Cartesian coordinates of this rank
    periodic::NTuple{3,Bool}
    nglob::NTuple{3,Int}
    nloc::NTuple{3,Int}               # local interior extents
    off::NTuple{3,Int}                # global index = off[d] + local index
    H::Int                            # halo width (active dimensions)
    active::NTuple{3,Bool}            # nglob[d] > 1; collapsed dims carry no
                                      # derivatives, halos, or exchanges
    Hd::NTuple{3,Int}                 # per-dim halo pad: H if active else 0
    nbr::NTuple{3,NTuple{2,Int}}      # (lo, hi) neighbor ranks per dim (MPI.PROC_NULL at open edges)
    sub::NTuple{3,MPI.Comm}           # sub-communicator along each dim
    subrank::NTuple{3,Int}
    subsize::NTuple{3,Int}
    sbuf::Vector{Vector{Float64}}     # per-dim halo send buffers
    rbuf::Vector{Vector{Float64}}     # per-dim halo recv buffers
end

"Even split of N points over P ranks; rank r (0-based) gets (count, offset)."
function local_range(N::Int, P::Int, r::Int)
    base, rem = divrem(N, P)
    n = base + (r < rem ? 1 : 0)
    off = r * base + min(r, rem)
    return n, off
end

function Decomp(nglob::NTuple{3,Int}, periodic::NTuple{3,Bool};
                dims=nothing, H::Int=4)
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    np = MPI.Comm_size(MPI.COMM_WORLD)
    active = ntuple(d -> nglob[d] > 1, 3)
    nact = count(active)
    if dims === nothing
        # MPI.Dims_create(1, k) is spec-legal (all ones) but MS-MPI rejects it,
        # so short-circuit the serial case where there is nothing to partition.
        adims = (nact == 0 || np == 1) ? fill(1, nact) : Int.(MPI.Dims_create(np, nact))
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
    nbr = ntuple(d -> Tuple(Int.(MPI.Cart_shift(comm, d - 1, 1))), 3)  # (src=lo, dst=hi)
    nl_off = ntuple(d -> local_range(nglob[d], pdims[d], coords[d]), 3)
    nloc = ntuple(d -> nl_off[d][1], 3)
    off  = ntuple(d -> nl_off[d][2], 3)
    sub = ntuple(d -> MPI.Cart_sub(comm, [k == d for k in 1:3]), 3)
    subrank = ntuple(d -> MPI.Comm_rank(sub[d]), 3)
    subsize = ntuple(d -> MPI.Comm_size(sub[d]), 3)
    Hd = ntuple(d -> active[d] ? H : 0, 3)
    full = ntuple(d -> nloc[d] + 2Hd[d], 3)
    cnt(d) = active[d] ? H * prod(full[k] for k in 1:3 if k != d) : 0
    sbuf = [zeros(cnt(d)) for d in 1:3]
    rbuf = [zeros(cnt(d)) for d in 1:3]
    Decomp(comm, pdims, coords, periodic, nglob, nloc, off, H, active, Hd,
           nbr, sub, subrank, subsize, sbuf, rbuf)
end

"CartesianIndices of the interior (halo-offset) block."
interior(dec::Decomp) =
    CartesianIndices(ntuple(d -> dec.Hd[d]+1:dec.Hd[d]+dec.nloc[d], 3))

"Allocate a scalar field with halos (collapsed dims carry no padding)."
field(dec::Decomp) = zeros(ntuple(d -> dec.nloc[d] + 2dec.Hd[d], 3))

"Allocate a conserved state Q(x,y,z,1:ncons) with halos."
allocate_state(dec::Decomp, ncons::Int) =
    zeros(ntuple(d -> dec.nloc[d] + 2dec.Hd[d], 3)..., ncons)

"Is this rank at the closed (non-periodic) global low/high edge of dim d?"
at_lo_edge(dec::Decomp, d::Int) =
    dec.active[d] && !dec.periodic[d] && dec.subrank[d] == 0
at_hi_edge(dec::Decomp, d::Int) =
    dec.active[d] && !dec.periodic[d] && dec.subrank[d] == dec.subsize[d] - 1
