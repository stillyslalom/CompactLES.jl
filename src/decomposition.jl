# 3-D MPI Cartesian domain decomposition.
#
# Global grid of n_global points per dimension is split evenly across a process
# grid `dims`. Each rank stores its interior block padded by n_halo halo layers
# on every side. Sub-communicators along each dimension support the distributed
# tridiagonal solves and line-wise collectives.

"""
    Decomp(n_global, periodic; dims=nothing, n_halo=4, comm=MPI.COMM_WORLD)

Three-dimensional Cartesian MPI decomposition of a structured global grid.
`dims` is the process grid; when omitted MPI chooses it over active dimensions,
and when given explicitly its product must equal the communicator size. Each
rank owns `n_local` interior points and halo-padded storage, `n_halo` layers on
each side of every active dimension. Dimensions with one global point are
collapsed: they are never decomposed, carry no halo pad, and take no part in an
exchange.

The constructor is collective over `comm` (default `MPI.COMM_WORLD`), and
initializes MPI at thread level `:funneled` if it is not yet initialized. It
builds a Cartesian communicator with reordering permitted, so a rank's Cartesian
coordinates need not follow its rank in `comm`, along with one sub-communicator
per dimension for the distributed line solves. On one rank of `MPI.COMM_WORLD`
or `MPI.COMM_SELF` it builds nothing: a single rank has no topology to arrange,
and those two communicators outlive any decomposition, so `comm` and every
`sub[d]` are that communicator itself, `owns_communicators` is false, and
`free_communicators!` has nothing to free. A patch-decomposed solver passes
the patch communicator produced by `MPI.Comm_split`; `n_global` is then the
patch extent, not the whole grid's.

The type parameter `T` is the element type of the halo exchange buffers, and
therefore of every field exchanged through them; the untyped constructor
defaults it to `Float64`.

Most users receive a `Decomp` through [`setup`](@ref). Construct one directly
only when using the low-level field and compact-operator APIs.
"""
struct Decomp{T}
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
    owns_communicators::Bool            # false on one rank of COMM_WORLD/COMM_SELF:
                                        # comm and sub are then that communicator
    send_buf::Vector{Vector{T}}         # per-dim halo send buffers
    recv_buf::Vector{Vector{T}}         # per-dim halo recv buffers
end

"""
    ConservedState(data)

Four-dimensional conserved-variable storage returned by [`allocate_state`](@ref)
and [`setup`](@ref). It behaves as an ordinary mutable `AbstractArray`; the
wrapper gives the state a concise REPL representation and avoids printing every
element after `setup` or [`run!`](@ref).

Use `parent(Q)` to access the underlying array when an external library
specifically requires a dense `Array`.
"""
struct ConservedState{T,A<:AbstractArray{T,4}} <: AbstractArray{T,4}
    data::A
end

Base.parent(Q::ConservedState) = Q.data
Base.size(Q::ConservedState) = size(parent(Q))
Base.axes(Q::ConservedState) = axes(parent(Q))
Base.IndexStyle(::Type{<:ConservedState{T,A}}) where {T,A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(Q::ConservedState, I...) =
    getindex(parent(Q), I...)
Base.@propagate_inbounds Base.setindex!(Q::ConservedState, value, I...) =
    setindex!(parent(Q), value, I...)

# Component views are the solver's common array-level operation. Returning a
# view of the dense parent keeps MPI packing and compact line solves on the same
# SubArray types they used before the display wrapper was introduced.
Base.view(Q::ConservedState, I...) = view(parent(Q), I...)
Base.copy(Q::ConservedState) = ConservedState(copy(parent(Q)))
Base.zero(Q::ConservedState) = ConservedState(zero(parent(Q)))
Base.copyto!(dest::ConservedState, src::ConservedState) =
    (copyto!(parent(dest), parent(src)); dest)
# Forward to the parent, bypassing the AbstractArray fallback, whose
# element-by-element loop is a scalar-indexing error on device storage.
Base.fill!(Q::ConservedState, x) = (fill!(parent(Q), x); Q)

"Balanced split of N points over P ranks; rank r (0-based) gets its
(count, 0-based offset). The first `N % P` ranks take one point more than the
rest, so counts differ by at most one and the offsets are contiguous."
function local_range(N::Int, P::Int, r::Int)
    base, rem = divrem(N, P)
    n = base + (r < rem ? 1 : 0)
    offset = r * base + min(r, rem)
    return n, offset
end

Decomp(n_global::NTuple{3,Int}, periodic::NTuple{3,Bool}; kwargs...) =
    Decomp{Float64}(n_global, periodic; kwargs...)

function Decomp{T}(n_global::NTuple{3,Int}, periodic::NTuple{3,Bool};
                   dims=nothing, n_halo::Int=4,
                   comm::MPI.Comm=MPI.COMM_WORLD) where {T}
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    np = MPI.Comm_size(comm)
    active = ntuple(d -> n_global[d] > 1, 3)
    nact = count(active)
    if dims === nothing
        # MPI.jl's second argument is the dims ARRAY (zeros = "choose for me"),
        # not the number of dimensions. Passing the count instead silently asks
        # for a 0-dimensional grid: it returns a 0-d array for nact ≤ 2 (which
        # popfirst! below cannot index) and throws MPIError(779) "cannot
        # partition nodes as requested" for nact == 3; that is, the default
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
    # One rank of a session-lifetime communicator borrows it. A Cartesian
    # topology over a single rank arranges nothing, and the four handles the
    # general path creates (the Cartesian communicator and its three
    # sub-communicators) are freed only from MPI.jl's finalizer, so a process
    # that builds and drops decompositions faster than the collector runs
    # exhausts MPICH's 2048 context ids: the serial test suite did, and a
    # level's refinement chains (eight `COMM_SELF` decompositions per tile,
    # held on every rank of the parent's subset) would at a few dozen tiles.
    # A split communicator of one rank is not borrowed, because its owner
    # frees it on its own schedule: a regrid drops a tile group and keeps the
    # tiles built on it.
    borrowed = np == 1 && (comm == MPI.COMM_WORLD || comm == MPI.COMM_SELF)
    if borrowed
        cart = comm
        coords = (0, 0, 0)
        # What MPI_Cart_shift returns on a one-rank dimension: the rank
        # itself where periodic, MPI_PROC_NULL where open.
        pnull = Int(MPI.PROC_NULL)
        neighbors = ntuple(d -> periodic[d] ? (0, 0) : (pnull, pnull), 3)
        sub = (comm, comm, comm)
    else
        cart = MPI.Cart_create(comm, collect(Int, pdims);
                               periodic=collect(periodic), reorder=true)
        coords = Tuple(Int.(MPI.Cart_coords(cart)))
        # (src=lo, dst=hi)
        neighbors = ntuple(d -> Tuple(Int.(MPI.Cart_shift(cart, d - 1, 1))), 3)
        # `sub_rank[d] == coords[d]` throughout the package: `at_lo_edge` and
        # `at_hi_edge` below, the fold pairing, the NSCBC wall tests and the
        # profile gathers in diagnostics.jl all read one where the other would
        # do. MPI_Cart_sub guarantees it: the sub-communicator along `d` is
        # ordered by the Cartesian coordinate in `d`, so a rank's position in
        # it is its own `coords[d]`. Nothing here re-derives the identity, so
        # a decomposition built by some route other than Cart_create/Cart_sub
        # would have to supply it (the borrowed branch does: one rank, at
        # coordinate 0 of every sub-communicator).
        sub = ntuple(d -> MPI.Cart_sub(cart, [k == d for k in 1:3]), 3)
    end
    nl_off = ntuple(d -> local_range(n_global[d], pdims[d], coords[d]), 3)
    n_local = ntuple(d -> nl_off[d][1], 3)
    offset  = ntuple(d -> nl_off[d][2], 3)
    sub_rank = ntuple(d -> MPI.Comm_rank(sub[d]), 3)
    sub_size = ntuple(d -> MPI.Comm_size(sub[d]), 3)
    n_halo_d = ntuple(d -> active[d] ? n_halo : 0, 3)
    full = ntuple(d -> n_local[d] + 2*n_halo_d[d], 3)
    cnt(d) = active[d] ? n_halo * prod(full[k] for k in 1:3 if k != d) : 0
    send_buf = [zeros(T, cnt(d)) for d in 1:3]
    recv_buf = [zeros(T, cnt(d)) for d in 1:3]
    Decomp{T}(cart, pdims, coords, periodic, n_global, n_local, offset, n_halo, active,
              n_halo_d, neighbors, sub, sub_rank, sub_size, !borrowed, send_buf,
              recv_buf)
end

"""
    free_communicators!(decomp)

Free the four communicators the constructor created: the Cartesian
communicator and its three sub-communicators. A no-op on a decomposition
that borrowed a one-rank `COMM_WORLD` or `COMM_SELF` and so created none
(`owns_communicators` is false). MPI.jl frees a communicator
only when garbage collection finalizes it, so a path that discards
decompositions in a loop (regridding rebuilds every level transfer's
refinement chains) exhausts MPICH's 2048-context-id budget whenever
collection lags, presenting as an intermittent `MPI_Cart_sub` "Too many
communicators" failure. Call this when a `Decomp` is permanently dropped;
the `Decomp` must not be used afterward. `MPI_Comm_free` is collective, so
every rank of the communicator must make the same call; the regrid paths
satisfy this because the tagged sets are reduced before any rank rebuilds,
and the chain decompositions, on `COMM_SELF`, own nothing. Freeing sets each
handle to `MPI_COMM_NULL` in place, which MPI.jl's finalizer guard then skips.
"""
function free_communicators!(decomp::Decomp)
    decomp.owns_communicators || return nothing
    for d in 1:3
        MPI.free(decomp.sub[d])
    end
    MPI.free(decomp.comm)
    return nothing
end

"""
    cart_rank(decomp, coords) -> Int

The rank in `decomp.comm` at the 0-based Cartesian coordinates `coords`:
`MPI.Cart_rank` where the constructor built a topology, and 0 on a borrowed
one-rank communicator, which carries none.
"""
cart_rank(decomp::Decomp, coords) =
    decomp.owns_communicators ? Int(MPI.Cart_rank(decomp.comm, Cint.(coords))) : 0

"CartesianIndices of the interior (halo-offset) block."
interior(decomp::Decomp) = CartesianIndices(
    ntuple(d -> decomp.n_halo_d[d]+1:decomp.n_halo_d[d]+decomp.n_local[d], 3))

"""
    AbstractBackend

Storage backend selecting where field arrays live. [`CPUBackend`](@ref) is the
default and allocates ordinary `Array`s; a GPU backend supplies device arrays
through the same [`field`](@ref) and [`allocate_state`](@ref) methods
(`reference/AMR_GPU.md`). Every solver array is typed
`A <: AbstractArray{T,3}` against this interface.
"""
abstract type AbstractBackend end

"""
    CPUBackend()

Host-memory backend: `field(CPUBackend(), decomp)` allocates a zero-filled
`Array{T,3}` for a `Decomp{T}`.
"""
struct CPUBackend <: AbstractBackend end

"The padded extent of one rank-local field, `n_local .+ 2 .* n_halo_d`: the
size [`field`](@ref) allocates, and the box a full-array pointwise launch
iterates (a stacked level's arrays span several such boxes)."
padded_extent(decomp::Decomp) =
    ntuple(d -> decomp.n_local[d] + 2 * decomp.n_halo_d[d], 3)

"Allocate a zero-filled `Array{T,3}` of extent `n_local .+ 2 .* n_halo_d`:
one rank-local scalar field with its halos, collapsed dimensions carrying no
padding. The one-argument form allocates on the CPU."
field(decomp::Decomp{T}) where {T} = zeros(T, padded_extent(decomp))
field(::CPUBackend, decomp::Decomp) = field(decomp)

"Zero-extent placeholder array on the backend's storage, for `Patch` fields a
configuration does not use (`pairbuf` without a fold, `ring_buf` under
`:delta4`). Typed like [`field`](@ref) so every array of a patch shares one
storage type."
empty_field(::CPUBackend, ::Type{T}) where {T} = zeros(T, 0, 0, 0)

"Allocate a zero-filled [`ConservedState`](@ref) Q(x,y,z,1:n_cons), with the
same spatial halo padding as [`field`](@ref)."
allocate_state(decomp::Decomp{T}, n_cons::Int) where {T} =
    ConservedState(zeros(T, ntuple(d -> decomp.n_local[d] + 2*decomp.n_halo_d[d], 3)...,
                         n_cons))
allocate_state(::CPUBackend, decomp::Decomp, n_cons::Int) =
    allocate_state(decomp, n_cons)

"""
    BlockRegion(offset, extent)

A rectangular index-space region of the global grid: a zero-based `offset` and
an `extent` per dimension, so the region covers global indices
`offset[d]+1 : offset[d]+extent[d]`. This is the unit both the patch layout and
the HDF5 hyperslab writes are expressed in, and it is distinct from a
`Decomp`: a region carries no communicator, no halo width and no process grid,
only where a block of data sits in the global index space.
"""
struct BlockRegion
    offset::NTuple{3,Int}
    extent::NTuple{3,Int}
end

"The `BlockRegion` covering this rank's interior block, in the index space of
the `Decomp`'s own grid (for a patch decomposition, the patch's)."
owned_region(decomp::Decomp) =
    BlockRegion(ntuple(d -> decomp.offset[d], 3), ntuple(d -> decomp.n_local[d], 3))

"Global index ranges of a [`BlockRegion`](@ref), one `UnitRange` per dimension."
region_ranges(region::BlockRegion) =
    ntuple(d -> (region.offset[d]+1):(region.offset[d]+region.extent[d]), 3)

"Is this rank at the closed (non-periodic) global low/high edge of dim d?"
at_lo_edge(decomp::Decomp, d::Int) =
    decomp.active[d] && !decomp.periodic[d] && decomp.sub_rank[d] == 0
at_hi_edge(decomp::Decomp, d::Int) =
    decomp.active[d] && !decomp.periodic[d] && decomp.sub_rank[d] == decomp.sub_size[d] - 1
