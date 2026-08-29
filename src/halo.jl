# Halo exchange.
#
# Dimensions are exchanged sequentially (x, then y, then z); each exchange
# includes the halo layers of previously exchanged dimensions in its slabs, so
# edge and corner halos are filled without dedicated diagonal messages. At
# non-periodic global edges the neighbor is MPI.PROC_NULL: the Sendrecv is a
# no-op on that side and the physical-edge halos are left stale.
#
# Device-resident fields stage each
# message through the backend: a broadcast pack of the strided slab into a
# contiguous device buffer, one contiguous device<->host copy per message, and
# the existing MPI path over the host buffers. Direct device-pointer MPI is
# not taken on any stack that does not report the capability for
# the active array type; no stack on the measured machines does, so the direct
# path is a future branch behind `device_mpi_direct`, not dead code today.
#
# At a CLOSED edge a stale halo is never read: the closure rows are applied to
# points counted inward from the edge, so a closure row indexes interior points
# by construction. A fold is the exception. There the interior stencil runs to
# the edge and does read the physical-edge halo, so `fold_fill!`
# writes the parity mirror into it before every folded sweep (folds.jl).
#
# --- Message tags ------------------------------------------------------------
#
# Three families of point-to-point message travel over `decomp.comm` and its
# sub-communicators, so their tags are allocated from disjoint ranges:
#
#   10d, 10d + 1     `exchange_dim!` and `_exchange_dim_staged!`, phase 1 and 2
#   100d, 100d + 1   `exchange_dim_batch!` and `_exchange_dim_batch_staged!`
#   41, 42           the fold pair exchanges in folds.jl, which reach the
#                    network through `sendrecv_block!`
#
# Disjointness is not the property correctness rests on. Every message here is
# a blocking `MPI.Sendrecv!` that all ranks of the sub-communicator reach in
# the same order, so at most one message per partner is in flight at a time and
# a receive cannot match a send from a later phase. The ranges are kept apart;
# a caller interleaving two families, or a future non-blocking variant, therefore
# does not depend on that argument. The patch interface records (patches.jl)
# tag from 2000 and 16000 over the world communicator and cannot reach these
# values.

const PNULL = Int(MPI.PROC_NULL)

@inline function _slab(f, d::Int, r::UnitRange{Int})
    view(f, ntuple(k -> k == d ? r : (1:size(f, k)), 3)...)
end

# --- Device staging ---------------------------------------------------------

"""
Test toggle: `true` routes the halo and fold-pair exchanges of ordinary
`Array` fields through the device staging path (pack buffers, offset copies,
and all), which on host storage performs the identical element copies. The
MPI suite flips this to pin the staging path bitwise on machines without a
GPU; `FORCE_KA` likewise pins the kernel path. The default `false` keeps
host fields on the direct slab copies.
"""
const FORCE_DEVICE_EXCHANGE = Ref(false)

@inline _staged_exchange(f) = !_cpu_storage(f) || FORCE_DEVICE_EXCHANGE[]

"""
    device_mpi_direct(backend) -> Bool

Whether MPI may receive this storage backend's device pointers directly. The
conservative default is `false`, which selects host staging; a device-aware
MPI stack can override this per backend once such a stack is measured. This
is a runtime property of the MPI library *and* the array type, so it is a
function of the backend, not a build-time switch.
"""
device_mpi_direct(backend) = false

# Staging buffers allocate per exchange through `similar`, whose result type
# is inferable from the field; a keyed cache was tried first and its
# `Any`-typed lookup put 33 runtime-dispatch sites into `compute_rhs!`'s
# jetcheck report, since the staged branch sits behind a runtime Ref read and
# is always inferred into the RHS call graph. Device allocators pool, so the
# per-exchange cost is a pool hit, not a device allocation; hoist them into
# retained storage only if a launch-level profile shows the pool hit at all.
@inline _device_stage(f, need::Int) =
    (similar(f, eltype(f), need), similar(f, eltype(f), need))

# Transfer accounting for the distributed-device measurements: bytes staged
# through the device
# buffers and the wall time of the synchronous copies, accumulated only while
# a bench turns the toggle on (the timer synchronizes nothing extra, since the
# copies are synchronous, but the clock calls are not free).
const TRACK_DEVICE_TRANSFERS = Ref(false)
const DEVICE_TRANSFER_BYTES = Ref(0)
const DEVICE_TRANSFER_TIME = Ref(0.0)

@inline function _tracked_copy!(dest, doffs::Int, src, soffs::Int, n::Int)
    if TRACK_DEVICE_TRANSFERS[]
        t0 = time_ns()
        copyto!(dest, doffs, src, soffs, n)
        DEVICE_TRANSFER_TIME[] += (time_ns() - t0) / 1e9
        DEVICE_TRANSFER_BYTES[] += n * sizeof(eltype(src))
    else
        copyto!(dest, doffs, src, soffs, n)
    end
    return dest
end

"""
    sendrecv_block!(send, recv, decomp, d, partner, tag)

Pairwise whole-array `MPI.Sendrecv!` used by the fold pairing: direct on host
storage, staged through dimension `d`'s host buffers on device storage (the
arrays are contiguous, so no pack kernel is needed: one copy each way).
Both partners must call it with the same tag.
"""
function sendrecv_block!(send, recv, decomp::Decomp, d::Int, partner::Int,
                         tag::Int)
    if _staged_exchange(send)
        n = length(send)
        sb, rb = decomp.send_buf[d], decomp.recv_buf[d]
        length(sb) < n && resize!(sb, n)
        length(rb) < n && resize!(rb, n)
        # Through the device stage even though the block is contiguous: the
        # broadcast handles wrapped storage (a component view of Q reaches
        # here from filter_state!), and the host copy then runs on the dense
        # stage vector, the one shape every backend's fast path covers.
        dsend, drecv = _device_stage(send, n)
        view(dsend, 1:n) .= vec(send)
        _tracked_copy!(sb, 1, dsend, 1, n)
        MPI.Sendrecv!(view(sb, 1:n), view(rb, 1:n), decomp.comm;
                      dest=partner, source=partner, sendtag=tag, recvtag=tag)
        _tracked_copy!(drecv, 1, rb, 1, n)
        vec(recv) .= view(drecv, 1:n)
    else
        MPI.Sendrecv!(send, recv, decomp.comm; dest=partner, source=partner,
                      sendtag=tag, recvtag=tag)
    end
    return recv
end

# Pack/unpack between a strided slab and the leading `slabsz` elements of a
# contiguous stage vector, as broadcasts so device storage runs them as
# kernels. `at` is a zero-based element offset into the stage vector.
@inline function _pack_stage!(stage, at::Int, f, d::Int, r::UnitRange{Int})
    s = _slab(f, d, r)
    reshape(view(stage, at+1:at+length(s)), size(s)) .= s
    return stage
end

@inline function _unpack_stage!(f, d::Int, r::UnitRange{Int}, stage, at::Int)
    s = _slab(f, d, r)
    s .= reshape(view(stage, at+1:at+length(s)), size(s))
    return f
end

"""
    selfwrap(decomp, d)

True when dimension `d` is periodic and undivided, so both neighbours are this
rank. The exchange is then a plain periodic copy of one slab onto another
within the same array, and routing it through pack / MPI.Sendrecv! / unpack
would add three passes over the slab and a self-message. This case also occurs
in decomposed calculations: with `dims = (P, 1, 1)`, every y and z exchange on
every rank uses this path.
"""
selfwrap(decomp::Decomp, d::Int) = decomp.sub_size[d] == 1 && decomp.periodic[d]

# One slab-onto-slab assignment inside a field: `copyto!` on host storage, a
# broadcast on device storage, since the generic `copyto!` fallback between two
# device views iterates scalar indices, which errors (or crawls) there. Both
# forms are element copies, so the result is identical.
@inline function _assign_slab!(f, d::Int, rdst::UnitRange{Int}, rsrc::UnitRange{Int})
    if _cpu_storage(f)
        copyto!(_slab(f, d, rdst), _slab(f, d, rsrc))
    else
        _slab(f, d, rdst) .= _slab(f, d, rsrc)
    end
    return f
end

"Periodic wrap of one field along `d`, in place, no buffers and no MPI."
function wrap_dim!(f, decomp::Decomp, d::Int)
    # The per-dimension pad, not `decomp.n_halo`: the two agree on an active
    # dimension, and a collapsed one has no pad at all.
    n_halo = decomp.n_halo_d[d]
    n = decomp.n_local[d]
    _assign_slab!(f, d, 1:n_halo,                  (n+1):(n+n_halo))
    _assign_slab!(f, d, (n+n_halo+1):(n+2*n_halo), (n_halo+1):(2*n_halo))
    return f
end

"""
    exchange_dim!(f, decomp, d)

Fill the rank-boundary halos of scalar field `f` along ONE dimension, in place,
and return `f`. A 1-D stencil along `d` needs no more, since corner halos are
read only by operators that reach off-axis; a caller applying a directional
operator (the compact filter in `smooth!`, for one) should take this route to
avoid paying for all three dimensions per direction.

A no-op when `d` is collapsed, and a buffer-free local copy when `d` is
undivided and periodic (see [`selfwrap`](@ref)). Otherwise every rank of the
sub-communicator along `d` must call it, from a serial section: the exchange is
two `MPI.Sendrecv!` phases through `decomp.send_buf[d]` and
`decomp.recv_buf[d]`, shared with [`exchange_dim_batch!`](@ref) and resized on
demand.

`f` must carry the `Decomp`'s own element type. The staging buffers are
`Vector{T}` for a `Decomp{T}`, so a field of any other type would be converted
into and back out of them one element at a time, losing precision on the way
out and costing a full pass over the slab in each direction. The signature
rejects that pairing instead.
"""
function exchange_dim!(f::AbstractArray{T,3}, decomp::Decomp{T},
                       d::Int) where {T<:Real}
    decomp.active[d] || return f
    selfwrap(decomp, d) && return wrap_dim!(f, decomp, d)
    _staged_exchange(f) && return _exchange_dim_staged!(f, decomp, d)
    let n_halo = decomp.n_halo
        n = decomp.n_local[d]
        lo, hi = decomp.neighbors[d]
        # The send/recv buffers are shared with exchange_dim_batch!, which grows
        # them to hold many components; use views sized to this field's slab so
        # only slab-many elements are transferred and copied.
        slabsz = n_halo * prod(size(f, k) for k in 1:3 if k != d)
        sb, rb = decomp.send_buf[d], decomp.recv_buf[d]
        length(sb) < slabsz && resize!(sb, slabsz)
        length(rb) < slabsz && resize!(rb, slabsz)
        sv = view(sb, 1:slabsz)
        rv = view(rb, 1:slabsz)
        # phase 1: send high-interior planes to hi neighbor, receive from lo
        hi != PNULL && copyto!(sv, _slab(f, d, (n+1):(n+n_halo)))
        MPI.Sendrecv!(sv, rv, decomp.comm; dest=hi, source=lo,
                      sendtag=10d, recvtag=10d)
        lo != PNULL && copyto!(_slab(f, d, 1:n_halo), rv)
        # phase 2: send low-interior planes to lo neighbor, receive from hi
        lo != PNULL && copyto!(sv, _slab(f, d, (n_halo+1):(2*n_halo)))
        MPI.Sendrecv!(sv, rv, decomp.comm; dest=lo, source=hi,
                      sendtag=10d + 1, recvtag=10d + 1)
        hi != PNULL && copyto!(_slab(f, d, (n+n_halo+1):(n+2*n_halo)), rv)
    end
    return f
end

# The staged counterpart of `exchange_dim!`: broadcast pack into the device
# stage, one contiguous copy each way around the same Sendrecv over the host
# buffers, broadcast unpack. Message tags and phase order are identical to the
# direct path, so mixed CPU/GPU rank sets would still pair up (not a
# supported configuration, but the property costs nothing to keep).
function _exchange_dim_staged!(f, decomp::Decomp, d::Int)
    n_halo = decomp.n_halo
    n = decomp.n_local[d]
    lo, hi = decomp.neighbors[d]
    slabsz = n_halo * prod(size(f, k) for k in 1:3 if k != d)
    sb, rb = decomp.send_buf[d], decomp.recv_buf[d]
    length(sb) < slabsz && resize!(sb, slabsz)
    length(rb) < slabsz && resize!(rb, slabsz)
    dsend, drecv = _device_stage(f, slabsz)
    sv = view(sb, 1:slabsz)
    rv = view(rb, 1:slabsz)
    # phase 1: send high-interior planes to hi neighbor, receive from lo
    if hi != PNULL
        _pack_stage!(dsend, 0, f, d, (n+1):(n+n_halo))
        _tracked_copy!(sb, 1, dsend, 1, slabsz)
    end
    MPI.Sendrecv!(sv, rv, decomp.comm; dest=hi, source=lo,
                  sendtag=10d, recvtag=10d)
    if lo != PNULL
        _tracked_copy!(drecv, 1, rb, 1, slabsz)
        _unpack_stage!(f, d, 1:n_halo, drecv, 0)
    end
    # phase 2: send low-interior planes to lo neighbor, receive from hi
    if lo != PNULL
        _pack_stage!(dsend, 0, f, d, (n_halo+1):(2*n_halo))
        _tracked_copy!(sb, 1, dsend, 1, slabsz)
    end
    MPI.Sendrecv!(sv, rv, decomp.comm; dest=lo, source=hi,
                  sendtag=10d + 1, recvtag=10d + 1)
    if hi != PNULL
        _tracked_copy!(drecv, 1, rb, 1, slabsz)
        _unpack_stage!(f, d, (n+n_halo+1):(n+2*n_halo), drecv, 0)
    end
    return f
end

"""
    exchange_halos!(f, decomp)

Fill the rank-boundary halos of scalar field `f` from neighboring ranks over all
three dimensions, in place, and return `f`. `f` must be sized
`n_local .+ 2 .* n_halo_d`, the size [`field`](@ref) allocates; collapsed
dimensions carry no pad and are skipped. Dimensions are exchanged in turn, so
edge and corner halos come out filled without dedicated diagonal messages.
Every rank must call it, from a serial (non-threaded) section. `f` carries the
`Decomp`'s element type, for the reason [`exchange_dim!`](@ref) gives.
"""
function exchange_halos!(f::AbstractArray{T,3}, decomp::Decomp{T}) where {T<:Real}
    for d in 1:3
        exchange_dim!(f, decomp, d)
    end
    return f
end

"""
Component views of the 4-D conserved array, indexed on demand and not
collected into a `Vector`. [`exchange_state!`](@ref) runs twice per step and
built that vector, and one `SubArray` per conserved component, on every
call; this wrapper is immutable, so the same loop allocates nothing.
"""
struct ComponentViews{V,A} <: AbstractVector{V}
    Q::A
end

# `typeof` of a view that is never built: the expression is only inferred.
ComponentViews(Q::AbstractArray{<:Any,4}) =
    ComponentViews{typeof(view(Q, :, :, :, 1)),typeof(Q)}(Q)

Base.size(c::ComponentViews) = (size(c.Q, 4),)
Base.IndexStyle(::Type{<:ComponentViews}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(c::ComponentViews, i::Int) =
    view(c.Q, :, :, :, i)

"Exchange halos of every conserved component of the 4-D state array in place,
returning `Q`. Components are batched into one message per neighbor per
dimension per phase, six per rank when all three dimensions are active and
decomposed. Every rank must call it, from a serial section."
function exchange_state!(Q::AbstractArray{T,4}, decomp::Decomp{T}) where {T<:Real}
    comps = ComponentViews(Q)
    for d in 1:3
        exchange_dim_batch!(comps, decomp, d)
    end
    return Q
end

@inline function _copy_interior_point!(dst, src, o1, o2, o3, i, j, k)
    @inbounds dst[i+o1, j+o2, k+o3] = src[i+o1, j+o2, k+o3]
    return nothing
end

"Copy the interior of `src` into `dst` (both full-sized fields), leaving the
halos of `dst` untouched, and return `dst`. A pointwise launch, rank-local,
no exchange."
function copy_interior!(dst, src, decomp::Decomp)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    pointwise!(_copy_interior_point!, dst, nx, ny, nz, dst, src, o1, o2, o3)
    return dst
end

"""
    blend_interior!(dst, src, w, decomp)

Interior-only relaxation `dst ← (1 − w) dst + w src`. The halos of `dst` are
left untouched and `dst` is returned.

`w = 1` reproduces `copy_interior!` exactly and is dispatched to it by the
caller, not computed here, so a full-strength application stays
bit-identical to the unrelaxed path.

`w` is used as given and is not reduced here, so a caller must supply the same
value on every rank; [`filter_weight`](@ref) builds it out of globally reduced
quantities for that reason.
"""
@inline function _blend_interior_point!(dst, src, w1, w, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        dst[I] = w1 * dst[I] + w * src[I]
    end
    return nothing
end

function blend_interior!(dst, src, w, decomp::Decomp)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    w1 = one(w) - w
    pointwise!(_blend_interior_point!, dst, nx, ny, nz, dst, src, w1, w,
               o1, o2, o3)
    return dst
end

"""
    exchange_dim_batch!(fields, decomp, d)

Exchange rank-boundary halos of many fields along a single dimension `d`, in
place, packed into one message per neighbor per phase. `fields` is any vector of
three-dimensional arrays or views of one common size, and is returned. Used for
flux arrays, which are differentiated only along their own direction and
therefore need halos only in that dimension, and for the conserved state (per
dim).

The slab size is taken from `fields[1]` and applied to all of them, as is the
element type, which must be the `Decomp`'s own for the reason
[`exchange_dim!`](@ref) gives. Buffers are grown on demand. As with
[`exchange_dim!`](@ref), every rank of the sub-communicator along `d` must call
this with the same number of fields, from a serial section.
"""
function exchange_dim_batch!(fields::AbstractVector, decomp::Decomp{T},
                             d::Int) where {T}
    isempty(fields) && return fields
    eltype(eltype(fields)) === T ||
        error("exchange_dim_batch!: fields carry $(eltype(eltype(fields))) " *
              "but the decomposition's exchange buffers are Vector{$T}")
    decomp.active[d] || return fields
    if selfwrap(decomp, d)
        for f in fields
            wrap_dim!(f, decomp, d)
        end
        return fields
    end
    _staged_exchange(fields[1]) &&
        return _exchange_dim_batch_staged!(fields, decomp, d)
    n_halo = decomp.n_halo
    n = decomp.n_local[d]
    lo, hi = decomp.neighbors[d]
    f1 = fields[1]
    slabsz = n_halo * prod(size(f1, k) for k in 1:3 if k != d)
    nf = length(fields)
    need = nf * slabsz
    sb, rb = decomp.send_buf[d], decomp.recv_buf[d]
    length(sb) < need && resize!(sb, need)
    length(rb) < need && resize!(rb, need)
    sv = view(sb, 1:need)
    rv = view(rb, 1:need)

    # phase 1: send high-interior planes to hi neighbor, receive from lo
    if hi != PNULL
        for (fi, f) in enumerate(fields)
            copyto!(sb, (fi - 1) * slabsz + 1, _slab(f, d, (n+1):(n+n_halo)), 1, slabsz)
        end
    end
    MPI.Sendrecv!(sv, rv, decomp.comm; dest=hi, source=lo,
                  sendtag=100d, recvtag=100d)
    if lo != PNULL
        for (fi, f) in enumerate(fields)
            copyto!(_slab(f, d, 1:n_halo), 1, rb, (fi - 1) * slabsz + 1, slabsz)
        end
    end
    # phase 2: send low-interior planes to lo neighbor, receive from hi
    if lo != PNULL
        for (fi, f) in enumerate(fields)
            copyto!(sb, (fi - 1) * slabsz + 1,
                    _slab(f, d, (n_halo+1):(2*n_halo)), 1, slabsz)
        end
    end
    MPI.Sendrecv!(sv, rv, decomp.comm; dest=lo, source=hi,
                  sendtag=100d + 1, recvtag=100d + 1)
    if hi != PNULL
        for (fi, f) in enumerate(fields)
            copyto!(_slab(f, d, (n+n_halo+1):(n+2*n_halo)), 1, rb,
                    (fi - 1) * slabsz + 1, slabsz)
        end
    end
    return fields
end

# Staged batch exchange: every field's slab packs into one device stage at
# its own offset, so each phase still costs one device-to-host copy, one
# Sendrecv, and one host-to-device copy however many components ride along.
function _exchange_dim_batch_staged!(fields::AbstractVector, decomp::Decomp,
                                     d::Int)
    n_halo = decomp.n_halo
    n = decomp.n_local[d]
    lo, hi = decomp.neighbors[d]
    f1 = fields[1]
    slabsz = n_halo * prod(size(f1, k) for k in 1:3 if k != d)
    nf = length(fields)
    need = nf * slabsz
    sb, rb = decomp.send_buf[d], decomp.recv_buf[d]
    length(sb) < need && resize!(sb, need)
    length(rb) < need && resize!(rb, need)
    dsend, drecv = _device_stage(f1, need)
    sv = view(sb, 1:need)
    rv = view(rb, 1:need)
    # phase 1: send high-interior planes to hi neighbor, receive from lo
    if hi != PNULL
        for (fi, f) in enumerate(fields)
            _pack_stage!(dsend, (fi - 1) * slabsz, f, d, (n+1):(n+n_halo))
        end
        _tracked_copy!(sb, 1, dsend, 1, need)
    end
    MPI.Sendrecv!(sv, rv, decomp.comm; dest=hi, source=lo,
                  sendtag=100d, recvtag=100d)
    if lo != PNULL
        _tracked_copy!(drecv, 1, rb, 1, need)
        for (fi, f) in enumerate(fields)
            _unpack_stage!(f, d, 1:n_halo, drecv, (fi - 1) * slabsz)
        end
    end
    # phase 2: send low-interior planes to lo neighbor, receive from hi
    if lo != PNULL
        for (fi, f) in enumerate(fields)
            _pack_stage!(dsend, (fi - 1) * slabsz, f, d,
                         (n_halo+1):(2*n_halo))
        end
        _tracked_copy!(sb, 1, dsend, 1, need)
    end
    MPI.Sendrecv!(sv, rv, decomp.comm; dest=lo, source=hi,
                  sendtag=100d + 1, recvtag=100d + 1)
    if hi != PNULL
        _tracked_copy!(drecv, 1, rb, 1, need)
        for (fi, f) in enumerate(fields)
            _unpack_stage!(f, d, (n+n_halo+1):(n+2*n_halo), drecv,
                           (fi - 1) * slabsz)
        end
    end
    return fields
end
