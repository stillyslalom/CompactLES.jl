# Pointwise-phase launch machinery (GPU Stage G1 of reference/AMR_GPU.md).
#
# Every pointwise phase of the solver — the loops that visit each grid point
# independently, as opposed to the compact line solves — is written as one
# per-point body function beside its driver and launched through one of two
# paths: `Array` storage takes the existing `@threaded` loop, which keeps the
# default CPU path bit-identical to the pre-G1 solver, and any other storage
# type launches a KernelAbstractions kernel on the backend its arrays belong
# to. `get_backend(::Array)` is the KA CPU backend, so tests and benches
# exercise the kernel path on ordinary arrays by calling [`pointwise_ka!`]
# directly; a device array reaches it automatically through [`pointwise!`].
#
# The bodies take plain arrays and scalars rather than the solver, which is
# most of what a device launch requires of them. The remaining device
# requirement is that every kernel argument adapt to an isbits form, and a
# `Vector{A}` or `Matrix{A}` does not — worse, it HANGS in kernel-argument
# adaptation rather than erroring (measured on AMDGPU). The field
# collections therefore reach the bodies as the [`FieldVector`](@ref) and
# [`FieldMatrix`](@ref) wrappers below — tuples of the same arrays, built
# once at `Patch` construction (`Patch.field_tuples`), indexed exactly as
# the `Vector`/`Matrix` forms are so the bodies read identically on either
# path — and the gas-model EOS objects adapt to coefficient mirrors at
# launch time (`physics.jl`). `Nasa9Mixture` has no mirror yet, per the
# plan's ordering: it needs the fixed-width interval table.

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
host it is a zero-cost wrapper — `getindex` forwards to the vector, so the
`@threaded` path indexes exactly what it always indexed — and its whole
purpose is to carry the Adapt rule: at device-kernel launch it adapts to a
[`DeviceFieldVector`](@ref), an isbits `NTuple` of the device-side arrays.
A bare `Vector` kernel argument instead *hangs* in kernel-argument
adaptation (measured on AMDGPU), which is why the bodies never see one.
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

Adapt.adapt_structure(to, w::FieldMatrix) =
    DeviceFieldMatrix{size(w.m, 1),length(w.m),
                      typeof(adapt(to, first(w.m)))}(
        map(x -> adapt(to, x), (w.m...,)))

"""
Test and benchmark toggle: `true` routes every [`pointwise!`](@ref) launch
through the KernelAbstractions path even on `Array` storage, where
`get_backend` supplies the KA CPU backend. The default `false` keeps `Array`
storage on the `@threaded` path, which is the configuration every guard was
measured under. The KA testset and `bench/pointwise_ka.jl` flip this around
their comparisons; nothing else should.
"""
const FORCE_KA = Ref(false)

"""
Launch synchronization policy (G3d): `false` (the default) defers — kernels
queue on the backend's one in-order stream and the host synchronizes only at
its interaction points (the reduced-solve staging fence, the synchronous
device↔host copies of the halo/ring staging and the reductions), which is
what removes the per-launch round trip the G1/G2 timings measured. `true`
restores the G3a conservative mode — a host synchronize after every
pointwise launch and between every line-solve subkernel — and is the
correctness fallback the plan requires: flip it when bisecting a device
discrepancy, so ordering bugs separate from arithmetic ones.
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
of the box in any order — the pointwise contract.
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
`KernelAbstractions.CPU()` — what `get_backend` returns for an `Array` —
this runs the same bodies through the kernel path on ordinary storage, which
is how the KA testset and `bench/pointwise_ka.jl` compare the two paths on
one machine.
"""
function pointwise_ka!(body!::F, backend, n1::Int, n2::Int, n3::Int,
                       args...) where {F}
    _point_kernel!(backend)(body!, args; ndrange=(n1, n2, n3))
    _maybe_sync(backend)
    return nothing
end
