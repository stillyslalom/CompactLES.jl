# Pointwise-phase launch machinery for the CPU and device backends. Design
# rationale and measurements: reference/AMR_GPU.md (pointwise kernels).
#
# Every pointwise phase of the solver (the loops that visit each grid point
# independently, as opposed to the compact line solves) is written as one
# per-point body function beside its driver and launched through one of two
# paths: `Array` storage takes the existing `@threaded` loop, the
# configuration every regression guard was measured under, and any other
# storage type launches a KernelAbstractions kernel on the backend its arrays
# belong to. `get_backend(::Array)` is the KA CPU backend, so tests and benches
# exercise the kernel path on ordinary arrays by calling [`pointwise_ka!`]
# directly; a device array reaches it automatically through [`pointwise!`].
#
# The bodies take plain arrays and scalars, not the solver, which is most of
# what a device launch requires of them. The remaining device requirement is
# that every kernel argument adapt to an isbits form, and a `Vector{A}` or
# `Matrix{A}` does not: it hangs in kernel-argument adaptation without an
# error (measured on AMDGPU). The field collections therefore reach the
# bodies as the [`FieldVector`](@ref) and [`FieldMatrix`](@ref) wrappers
# below, tuples of the same arrays, built once at `Patch` construction
# (`Patch.field_tuples`) and using the same indexing as the `Vector`/`Matrix`
# forms, so the bodies read identically on either path; the gas-model EOS
# objects adapt to coefficient mirrors at launch time (`physics.jl`).
# `Nasa9Mixture` has no mirror yet: it needs the fixed-width interval table
# (reference/AMR_GPU.md, open numerics).

using KernelAbstractions
using KernelAbstractions: get_backend, synchronize
using Adapt

# The routing test, seeing through the ConservedState display wrapper and
# SubArray views: `Array` storage takes the @threaded path, anything else the
# KA kernel path. Each method resolves statically for a concrete storage type.
@inline _cpu_storage(::Array) = true
@inline _cpu_storage(Q::ConservedState) = _cpu_storage(parent(Q))
@inline _cpu_storage(f::SubArray) = _cpu_storage(parent(f))
@inline _cpu_storage(::AbstractArray) = false

KernelAbstractions.get_backend(Q::ConservedState) = get_backend(parent(Q))

# A conserved state reaches device kernels as the same wrapper around the
# adapted (device-side) array, so the bodies index `Q[I, c]` unchanged. Without
# this rule the wrapper would carry the host-side device-array object into the
# kernel, which does not lower.
Adapt.adapt_structure(to, Q::ConservedState) = ConservedState(adapt(to, parent(Q)))

"""
    FieldVector(v::AbstractVector)

Launchable stand-in for a `Vector` of field arrays (`Y`, `D_art`). On the
host it is a zero-cost wrapper: `getindex` forwards to the vector, so the
`@threaded` path indexes the same fields as before. Its purpose
is to carry the Adapt rule: at device-kernel launch it adapts to a
[`DeviceFieldVector`](@ref), an isbits `NTuple` of the device-side arrays.
A bare `Vector` kernel argument instead *hangs* in kernel-argument
adaptation (measured on AMDGPU), so the bodies never see one.
An early version held the `NTuple` on the host too; runtime tuple indexing
in the species loops cost `assemble_fluxes!` 3× on the `@threaded` path,
so the tuple now materializes only at launch. Built once per patch
(`Patch.field_tuples`).
"""
struct FieldVector{A,V<:AbstractVector{A}}
    v::V
end

Base.@propagate_inbounds Base.getindex(w::FieldVector, i::Int) = w.v[i]
Base.length(w::FieldVector) = length(w.v)

"""
    DeviceFieldVector

The device-side form of a [`FieldVector`](@ref): the same arrays as an
isbits `NTuple`, indexed as the vector was. Constructed by Adapt at kernel
launch; never on the host path.
"""
struct DeviceFieldVector{N,A}
    data::NTuple{N,A}
end

Base.@propagate_inbounds Base.getindex(w::DeviceFieldVector, i::Int) = w.data[i]
Base.length(::DeviceFieldVector{N}) where {N} = N

Adapt.adapt_structure(to, w::FieldVector) =
    DeviceFieldVector(map(x -> adapt(to, x), (w.v...,)))

"""
    FieldMatrix(m::AbstractMatrix)

The two-index counterpart of [`FieldVector`](@ref) for `grad_u`, `grad_Y`
and `flux`: a zero-cost host wrapper of the `Matrix` that adapts to a
[`DeviceFieldMatrix`](@ref) at device-kernel launch, so `grad_u[a, b][I]`
reads unchanged on either path.
"""
struct FieldMatrix{A,M<:AbstractMatrix{A}}
    m::M
end

Base.@propagate_inbounds Base.getindex(w::FieldMatrix, a::Int, b::Int) = w.m[a, b]
Base.size(w::FieldMatrix) = size(w.m)

"The device-side form of a [`FieldMatrix`](@ref): column-major `NTuple`."
struct DeviceFieldMatrix{N1,L,A}
    data::NTuple{L,A}
end

Base.@propagate_inbounds Base.getindex(w::DeviceFieldMatrix{N1}, a::Int,
                                       b::Int) where {N1} =
    w.data[(b - 1) * N1 + a]
Base.size(::DeviceFieldMatrix{N1,L}) where {N1,L} = (N1, L ÷ N1)

Adapt.adapt_structure(to, w::FieldMatrix) =
    DeviceFieldMatrix{size(w.m, 1),length(w.m),
                      typeof(adapt(to, first(w.m)))}(
        map(x -> adapt(to, x), (w.m...,)))

"""
    StackedArray(data, ntiles, stride)

The stacked storage of a device level's tiles (reference/AMR_GPU.md, launch
policy): one array holding every tile's padded block along the third
dimension at a fixed `stride`, the padded extent of one tile. A tile's
`Patch` holds a plain view of `data` over its own block, so the per-tile
paths (the impositions, the interface records, the gathers, `max_rate`) see
ordinary storage; the level's spanning patch holds this wrapper, and a
[`pointwise!`](@ref) launch routed on it, or a `DevicePlan` built for it,
runs every tile in one index space. The wrapper forwards indexing to `data`
and adapts to `data` itself at a device launch, so a body indexes the
stacked array with its own padded arithmetic and lands in tile `t` through
the `(t − 1)·stride` the launcher adds to `k`. `view` unwraps to a view of
`data`, except the component view `(:, :, :, c)` of a 4-D stack, which keeps
the wrapper so the filter's per-component launches stay batched.
"""
struct StackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    ntiles::Int
    stride::Int
end

Base.parent(s::StackedArray) = s.data
Base.size(s::StackedArray) = size(s.data)
Base.axes(s::StackedArray) = axes(s.data)
Base.IndexStyle(::Type{StackedArray{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(s::StackedArray, I...) = getindex(s.data, I...)
Base.@propagate_inbounds Base.setindex!(s::StackedArray, v, I...) =
    setindex!(s.data, v, I...)
Base.view(s::StackedArray, I...) = view(s.data, I...)
Base.view(s::StackedArray{T,4}, ::Colon, ::Colon, ::Colon, c::Int) where {T} =
    StackedArray(view(s.data, :, :, :, c), s.ntiles, s.stride)
# Forwarded past the AbstractArray fallbacks, which index element by element.
Base.fill!(s::StackedArray, x) = (fill!(s.data, x); s)
Base.similar(s::StackedArray, ::Type{S}, dims::Dims) where {S} = similar(s.data, S, dims)
Base.show(io::IO, s::StackedArray) =
    print(io, "StackedArray(", size(s), ", ", s.ntiles, " tiles, stride ", s.stride, ")")
Base.show(io::IO, ::MIME"text/plain", s::StackedArray) = show(io, s)
KernelAbstractions.get_backend(s::StackedArray) = get_backend(s.data)
Adapt.adapt_structure(to, s::StackedArray) = adapt(to, s.data)
@inline _cpu_storage(s::StackedArray) = _cpu_storage(s.data)

# The stacking of an array, `nothing` for ordinary storage. A tile's view of a
# stack is ordinary storage: the per-tile paths launch flat over it.
@inline _stack_of(::AbstractArray) = nothing
@inline _stack_of(s::StackedArray) = (s.ntiles, s.stride)
@inline _stack_of(Q::ConservedState) = _stack_of(parent(Q))

"The third-dimension range of each tile's block in a stack, in slot order."
_tile_ranges(ntiles::Int, stride::Int) =
    [((t - 1) * stride + 1):(t * stride) for t in 1:ntiles]
_tile_ranges(s::StackedArray) = _tile_ranges(s.ntiles, s.stride)

"""
Test and benchmark toggle: `true` routes every [`pointwise!`](@ref) launch
through the KernelAbstractions path even on `Array` storage, where
`get_backend` supplies the KA CPU backend. The default `false` keeps `Array`
storage on the `@threaded` path, which is the configuration every guard was
measured under. The device testsets flip this around the runs they compare
against the `CPUBackend` bitwise; nothing else should.
"""
const FORCE_KA = Ref(false)

"""
Launch synchronization policy: `false` (the default) defers. Kernels queue
on the backend's one in-order stream and the host synchronizes only at its
interaction points (the reduced-solve staging fence, the synchronous
device↔host copies of the halo/ring staging and the reductions); this
removes the per-launch round trip measured at 28–32% of a device step
(reference/AMR_GPU.md, launch policy). `true` restores the conservative
mode, a host synchronize after every pointwise launch and between every
line-solve subkernel, and is the correctness fallback: flip it when
bisecting a device discrepancy, so ordering bugs separate from arithmetic
ones.
"""
const DEVICE_SYNC = Ref(false)

@inline _maybe_sync(backend) = (DEVICE_SYNC[] && synchronize(backend); nothing)

@kernel function _point_kernel!(body!, args)
    i, j, k = @index(Global, NTuple)
    body!(args..., i, j, k)
end

"""
    pointwise!(body!, route, n1, n2, n3, args...)

Run `body!(args..., i, j, k)` over the `(1:n1) × (1:n2) × (1:n3)` index box.
`route` is a representative field array: `Array` storage takes the
`@threaded` loop over [`outer_indices`](@ref), anything else a
KernelAbstractions kernel on `get_backend(route)`. The branch resolves
statically for a concrete storage type. The body owns its own `@inbounds`
and any halo offsets, so the index box is the body's iteration count, not
necessarily a padded extent, and every body must be safe to run at any point
of the box in any order; this is the pointwise contract. A route that is a
`StackedArray` (or a conserved state over one) runs the box once per tile in
one launch; see the stacked launches below.
"""
@inline function pointwise!(body!::F, route::AbstractArray, n1::Int, n2::Int,
                            n3::Int, args...) where {F}
    if _cpu_storage(route) && !FORCE_KA[]
        @threaded n1 * n2 * n3 for jk in outer_indices(n2, n3)
            j, k = Tuple(jk)
            for i in 1:n1
                body!(args..., i, j, k)
            end
        end
        return nothing
    end
    return pointwise_ka!(body!, get_backend(route), n1, n2, n3, args...)
end

"""
    pointwise_ka!(body!, backend, n1, n2, n3, args...)

The KernelAbstractions launch of [`pointwise!`](@ref) on an explicit
`backend`. Under the default deferred policy the launch queues on the
backend's in-order stream and returns; [`DEVICE_SYNC`](@ref) restores the
synchronize-per-launch conservative mode. Called with
`KernelAbstractions.CPU()`, the backend `get_backend` returns for an `Array`,
this runs the same bodies through the kernel path on ordinary storage, which
is how `bench/pointwise_ka.jl` times the two paths against each other on one
machine. The test suite uses [`FORCE_KA`](@ref) to route a whole run through the
same path.
"""
function pointwise_ka!(body!::F, backend, n1::Int, n2::Int, n3::Int,
                       args...) where {F}
    _point_kernel!(backend)(body!, map(_kernel_arg, args); ndrange=(n1, n2, n3))
    _maybe_sync(backend)
    return nothing
end

# The form of a body argument inside a kernel: a conserved state or a
# stacked array hands over the array it wraps, since the bodies index both
# exactly as they index the array. A device launch would adapt the wrappers
# to isbits forms anyway; on the KernelAbstractions CPU backend, which adapts
# nothing, a wrapper's forwarding `getindex` is not inlined into the kernel
# and its Vararg dispatch allocates once per point (measured at one
# allocation per point for either wrapper), so the unwrapping here is what
# keeps the forced-KA test path at the speed of the device path. The
# `@threaded` path passes the wrappers through, inlined, as it always has.
@inline _kernel_arg(x) = x
@inline _kernel_arg(s::StackedArray) = s.data
@inline _kernel_arg(Q::ConservedState) = _kernel_arg(parent(Q))

# --- Stacked launches -------------------------------------------------------
#
# A launch routed on a `StackedArray` runs the same `(1:n1) × (1:n2) × (1:n3)`
# box once per tile, as one index space with a fourth dimension over the
# tiles, and hands the body `k + (t − 1)·stride`: the body's own padded
# offset along the third dimension then lands in tile t's block. The
# per-point arithmetic is the flat launch's, so a stacked level reproduces
# its tiles' separate launches bitwise; what changes is one launch per phase
# per level in place of one per tile. `n3` is a per-tile count and may not
# exceed the stride; a body that reads its `k` as a logical coordinate
# rather than an array offset (`_delta4_point!`'s edge clamp) must reduce it
# modulo the stride, as that body does.

@kernel function _point_kernel_stacked!(body!, args, stride)
    i, j, k, t = @index(Global, NTuple)
    body!(args..., i, j, k + (t - 1) * stride)
end

@inline pointwise!(body!::F, route::StackedArray, n1::Int, n2::Int, n3::Int,
                   args...) where {F} =
    _pointwise_stacked!(body!, route.data, n1, n2, n3, route.ntiles, route.stride,
                        args...)
@inline pointwise!(body!::F, route::ConservedState{T,<:StackedArray}, n1::Int,
                   n2::Int, n3::Int, args...) where {F,T} =
    pointwise!(body!, parent(route), n1, n2, n3, args...)

@noinline _stack_extent_error(n3::Int, stride::Int) =
    error("stacked launch over $n3 points along the stacking dimension exceeds " *
          "the tile stride $stride; launch over the tile's own extent")

@inline function _pointwise_stacked!(body!::F, data::AbstractArray, n1::Int, n2::Int,
                                     n3::Int, ntiles::Int, stride::Int,
                                     args...) where {F}
    n3 <= stride || _stack_extent_error(n3, stride)
    raw = map(_kernel_arg, args)
    if _cpu_storage(data) && !FORCE_KA[]
        @threaded n1 * n2 * n3 * ntiles for jkt in CartesianIndices((n2, n3, ntiles))
            j, k, t = Tuple(jkt)
            ks = k + (t - 1) * stride
            for i in 1:n1
                body!(raw..., i, j, ks)
            end
        end
        return nothing
    end
    backend = get_backend(data)
    _point_kernel_stacked!(backend)(body!, raw, stride; ndrange=(n1, n2, n3, ntiles))
    _maybe_sync(backend)
    return nothing
end
