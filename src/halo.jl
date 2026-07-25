# Halo exchange.
#
# Dimensions are exchanged sequentially (x, then y, then z); each exchange
# includes the halo layers of previously exchanged dimensions in its slabs, so
# edge and corner halos are filled without dedicated diagonal messages. At
# non-periodic global edges the neighbor is MPI.PROC_NULL: the Sendrecv is a
# no-op on that side and the physical-edge halos are left stale. They are never
# read, because derivative closures are one-sided and filter closure rows are
# identities near closed edges.

const PNULL = Int(MPI.PROC_NULL)

@inline function _slab(f, d::Int, r::UnitRange{Int})
    view(f, ntuple(k -> k == d ? r : (1:size(f, k)), 3)...)
end

"""
    selfwrap(dec, d)

True when dimension `d` is periodic and undivided, so both neighbours are this
rank. The exchange is then a plain periodic copy of one slab onto another
within the same array, and routing it through pack / MPI.Sendrecv! / unpack
costs three extra passes over the slab plus a self-message. This is not just
the serial case: with `dims = (P, 1, 1)` every y and z exchange on every rank
takes this path.
"""
selfwrap(dec::Decomp, d::Int) = dec.subsize[d] == 1 && dec.periodic[d]

"Periodic wrap of one field along `d`, in place, no buffers and no MPI."
function wrap_dim!(f, dec::Decomp, d::Int)
    H = dec.H
    n = dec.nloc[d]
    copyto!(_slab(f, d, 1:H),               _slab(f, d, (n+1):(n+H)))
    copyto!(_slab(f, d, (n+H+1):(n+2H)),    _slab(f, d, (H+1):(2H)))
    return f
end

"""
    exchange_dim!(f, dec, d)

Fill the rank-boundary halos of scalar field `f` along ONE dimension. This is
all a 1-D stencil along `d` needs: corner halos only matter to operators that
read off-axis, so anything applying a directional operator (the compact filter
in `smooth!`, for one) should call this rather than paying for all three
dimensions per direction.
"""
function exchange_dim!(f::AbstractArray{<:Real,3}, dec::Decomp, d::Int)
    dec.active[d] || return f
    selfwrap(dec, d) && return wrap_dim!(f, dec, d)
    let H = dec.H
        n = dec.nloc[d]
        lo, hi = dec.nbr[d]
        # The send/recv buffers are shared with exchange_dim_batch!, which grows
        # them to hold many components; use views sized to this field's slab so
        # only slab-many elements are transferred and copied.
        slabsz = H * prod(size(f, k) for k in 1:3 if k != d)
        sb, rb = dec.sbuf[d], dec.rbuf[d]
        length(sb) < slabsz && resize!(sb, slabsz)
        length(rb) < slabsz && resize!(rb, slabsz)
        sv = view(sb, 1:slabsz)
        rv = view(rb, 1:slabsz)
        # phase 1: send high-interior planes to hi neighbor, receive from lo
        hi != PNULL && copyto!(sv, _slab(f, d, (n+1):(n+H)))
        MPI.Sendrecv!(sv, rv, dec.comm; dest=hi, source=lo,
                      sendtag=10d, recvtag=10d)
        lo != PNULL && copyto!(_slab(f, d, 1:H), rv)
        # phase 2: send low-interior planes to lo neighbor, receive from hi
        lo != PNULL && copyto!(sv, _slab(f, d, (H+1):(2H)))
        MPI.Sendrecv!(sv, rv, dec.comm; dest=lo, source=hi,
                      sendtag=10d + 1, recvtag=10d + 1)
        hi != PNULL && copyto!(_slab(f, d, (n+H+1):(n+2H)), rv)
    end
    return f
end

"""
    exchange_halos!(f, dec)

Fill the rank-boundary halos of scalar field `f` (sized nloc .+ 2H) from
neighboring ranks, all three dimensions. Sequential per dimension, so edge and
corner halos come out filled without dedicated diagonal messages. Must be
called from a serial (non-threaded) section.
"""
function exchange_halos!(f::AbstractArray{<:Real,3}, dec::Decomp)
    for d in 1:3
        exchange_dim!(f, dec, d)
    end
    return f
end

"Exchange halos of every conserved component of the 4-D state array, batched
into one message per neighbor per dimension per phase (six messages total)."
function exchange_state!(Q::AbstractArray{<:Real,4}, dec::Decomp)
    comps = [view(Q, :, :, :, c) for c in 1:size(Q, 4)]
    for d in 1:3
        exchange_dim_batch!(comps, dec, d)
    end
    return Q
end

"Copy the interior of `src` into `dst` (both full-sized fields)."
function copy_interior!(dst, src, dec::Decomp)
    o1, o2, o3 = dec.Hd
    @inbounds @threaded prod(dec.nloc) for k in o3+1:o3+dec.nloc[3]
        for j in o2+1:o2+dec.nloc[2], i in o1+1:o1+dec.nloc[1]
            dst[i, j, k] = src[i, j, k]
        end
    end
    return dst
end

"""
    axis_fill!(f, dec, σf)

Parity fill of the low-dim-1 halos for the cylindrical-axis treatment on a
half-offset grid (r_i = (i − ½)h): ghost layer j mirrors interior layer j
with sign `σf` (+1 even fields, −1 odd). No-op unless this rank owns the low
edge of dimension 1.
"""
function axis_fill!(f, dec::Decomp, σf::Int)
    (dec.active[1] && !dec.periodic[1] && dec.subrank[1] == 0) || return f
    H1 = dec.Hd[1]
    s = Float64(σf)
    @inbounds for j in 1:H1
        g = H1 - j + 1          # ghost full index
        m = H1 + j              # mirrored interior full index
        @views f[g, :, :] .= s .* f[m, :, :]
    end
    return f
end

"""
    exchange_dim_batch!(fields, dec, d)

Exchange rank-boundary halos of many same-sized fields along a single
dimension `d`, packed into one message per neighbor per phase. Used for flux
arrays, which are differentiated only along their own direction and therefore
need halos only in that dimension, and for the conserved state (per dim).
Buffers are grown on demand; MPI calls run from serial sections only.
"""
function exchange_dim_batch!(fields::AbstractVector, dec::Decomp, d::Int)
    isempty(fields) && return fields
    dec.active[d] || return fields
    if selfwrap(dec, d)
        for f in fields
            wrap_dim!(f, dec, d)
        end
        return fields
    end
    H = dec.H
    n = dec.nloc[d]
    lo, hi = dec.nbr[d]
    f1 = fields[1]
    slabsz = H * prod(size(f1, k) for k in 1:3 if k != d)
    nf = length(fields)
    need = nf * slabsz
    sb, rb = dec.sbuf[d], dec.rbuf[d]
    length(sb) < need && resize!(sb, need)
    length(rb) < need && resize!(rb, need)
    sv = view(sb, 1:need)
    rv = view(rb, 1:need)

    # phase 1: send high-interior planes to hi neighbor, receive from lo
    if hi != PNULL
        for (fi, f) in enumerate(fields)
            copyto!(sb, (fi - 1) * slabsz + 1, _slab(f, d, (n+1):(n+H)), 1, slabsz)
        end
    end
    MPI.Sendrecv!(sv, rv, dec.comm; dest=hi, source=lo,
                  sendtag=20d, recvtag=20d)
    if lo != PNULL
        for (fi, f) in enumerate(fields)
            copyto!(_slab(f, d, 1:H), 1, rb, (fi - 1) * slabsz + 1, slabsz)
        end
    end
    # phase 2: send low-interior planes to lo neighbor, receive from hi
    if lo != PNULL
        for (fi, f) in enumerate(fields)
            copyto!(sb, (fi - 1) * slabsz + 1, _slab(f, d, (H+1):(2H)), 1, slabsz)
        end
    end
    MPI.Sendrecv!(sv, rv, dec.comm; dest=lo, source=hi,
                  sendtag=20d + 1, recvtag=20d + 1)
    if hi != PNULL
        for (fi, f) in enumerate(fields)
            copyto!(_slab(f, d, (n+H+1):(n+2H)), 1, rb, (fi - 1) * slabsz + 1, slabsz)
        end
    end
    return fields
end
