# Halo exchange.
#
# Dimensions are exchanged sequentially (x, then y, then z); each exchange
# includes the halo layers of previously exchanged dimensions in its slabs, so
# edge and corner halos are filled without dedicated diagonal messages. At
# non-periodic global edges the neighbor is MPI.PROC_NULL: the Sendrecv is a
# no-op on that side and the physical-edge halos are left stale.
#
# At a CLOSED edge a stale halo is never read: the closure rows are applied to
# points counted inward from the edge, so a closure row indexes interior points
# by construction. A fold is the exception. There the interior stencil runs to
# the edge and does read the physical-edge halo, which is why `fold_fill!`
# writes the parity mirror into it before every folded sweep (folds.jl).

const PNULL = Int(MPI.PROC_NULL)

@inline function _slab(f, d::Int, r::UnitRange{Int})
    view(f, ntuple(k -> k == d ? r : (1:size(f, k)), 3)...)
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
# broadcast on device storage — the generic `copyto!` fallback between two
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
    n_halo = decomp.n_halo
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
operator (the compact filter in `smooth!`, for one) should take this route
rather than paying for all three dimensions per direction.

A no-op when `d` is collapsed, and a buffer-free local copy when `d` is
undivided and periodic (see [`selfwrap`](@ref)). Otherwise every rank of the
sub-communicator along `d` must call it, from a serial section: the exchange is
two `MPI.Sendrecv!` phases through `decomp.send_buf[d]` and
`decomp.recv_buf[d]`, shared with [`exchange_dim_batch!`](@ref) and resized on
demand.
"""
function exchange_dim!(f::AbstractArray{<:Real,3}, decomp::Decomp, d::Int)
    decomp.active[d] || return f
    selfwrap(decomp, d) && return wrap_dim!(f, decomp, d)
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

"""
    exchange_halos!(f, decomp)

Fill the rank-boundary halos of scalar field `f` from neighboring ranks over all
three dimensions, in place, and return `f`. `f` must be sized
`n_local .+ 2 .* n_halo_d`, which is what [`field`](@ref) allocates; collapsed
dimensions carry no pad and are skipped. Dimensions are exchanged in turn, so
edge and corner halos come out filled without dedicated diagonal messages.
Every rank must call it, from a serial (non-threaded) section.
"""
function exchange_halos!(f::AbstractArray{<:Real,3}, decomp::Decomp)
    for d in 1:3
        exchange_dim!(f, decomp, d)
    end
    return f
end

"Exchange halos of every conserved component of the 4-D state array in place,
returning `Q`. Components are batched into one message per neighbor per
dimension per phase, six per rank when all three dimensions are active and
decomposed. Every rank must call it, from a serial section."
function exchange_state!(Q::AbstractArray{<:Real,4}, decomp::Decomp)
    comps = [view(Q, :, :, :, c) for c in 1:size(Q, 4)]
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
caller rather than computed here, so a full-strength application stays
bit-identical to the unrelaxed path.

`w` is used as given and is not reduced here, so a caller must supply the same
value on every rank; [`filter_weight`](@ref) builds it out of already-reduced
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

The slab size is taken from `fields[1]` and applied to all of them. Buffers are
grown on demand. As with [`exchange_dim!`](@ref), every rank of the
sub-communicator along `d` must call this with the same number of fields, from a
serial section.
"""
function exchange_dim_batch!(fields::AbstractVector, decomp::Decomp, d::Int)
    isempty(fields) && return fields
    decomp.active[d] || return fields
    if selfwrap(decomp, d)
        for f in fields
            wrap_dim!(f, decomp, d)
        end
        return fields
    end
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
                  sendtag=20d, recvtag=20d)
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
                  sendtag=20d + 1, recvtag=20d + 1)
    if hi != PNULL
        for (fi, f) in enumerate(fields)
            copyto!(_slab(f, d, (n+n_halo+1):(n+2*n_halo)), 1, rb,
                    (fi - 1) * slabsz + 1, slabsz)
        end
    end
    return fields
end
