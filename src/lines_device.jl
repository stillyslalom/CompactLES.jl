# Device path for the compact line solves. Design rationale and
# measurements: reference/AMR_GPU.md (line solves).
#
# The distributed compact solve is sequential along a line and independent
# across lines, so the device mapping is one thread per line for the
# elimination sweeps and one thread per point for the fill, scatter, and
# spike-correction passes. Every dimension uses the (lines × n) layout on
# device: consecutive threads are consecutive lines, so each sweep position
# touches a contiguous block of memory, the same reasoning that produced the
# transposed host path in lines_transposed.jl, with warp coalescing in place
# of SIMD.
#
# The reduced 2qP × 2qP interface system stays host-side: the
# device packs the 2q interface values per line into a buffer whose layout
# matches `line_solver.ends`, one device-to-host copy feeds the existing
# `_reduced_solve!` (Allgather + factorized `ldiv!`), and one host-to-device
# copy returns the per-line correction values. That transfer is 4q values per
# line and step, small next to the field itself; move it on-device only if
# profiling shows it dominating.
#
# Arithmetic mirrors the host path per line, operation for operation, so a
# [`DevicePlan`](@ref) reproduces its host plan bitwise, the same equality
# gate the pointwise kernels carry. One place in the banded solve the two host
# layouts still disagree: the x-sweep accumulates the spike correction before
# one subtraction where the transposed sweeps subtract term by term, and
# `colwise` selects the convention matching the plan's dimension. The
# elimination sweep and the back-substitution pivot were once such places too
# and are no longer: every host path now skips a zero multiplier and
# multiplies by `inv(U[1, i])`, so the device kernel does both unconditionally.

"""
    DeviceBackend(ka)

Storage backend wrapping a `KernelAbstractions.Backend`. This makes
[`field`](@ref) and [`allocate_state`](@ref) allocate zero-filled device
arrays through the same interface `CPUBackend` allocates `Array`s. A whole
`Solver` runs on one: [`backend_plan`](@ref) mirrors every compact plan onto
the backend at construction, the pointwise phases launch as
KernelAbstractions kernels, and the halo, fold-pair and level-transfer
exchanges take their staged forms. The configurations refused at setup are
listed with the `Solver` constructor.
"""
struct DeviceBackend{B} <: AbstractBackend
    ka::B
end

field(backend::DeviceBackend, decomp::Decomp{T}) where {T} =
    KernelAbstractions.zeros(backend.ka, T,
        ntuple(d -> decomp.n_local[d] + 2*decomp.n_halo_d[d], 3))

allocate_state(backend::DeviceBackend, decomp::Decomp{T}, n_cons::Int) where {T} =
    ConservedState(KernelAbstractions.zeros(backend.ka, T,
        ntuple(d -> decomp.n_local[d] + 2*decomp.n_halo_d[d], 3)..., n_cons))

empty_field(backend::DeviceBackend, ::Type{T}) where {T} =
    KernelAbstractions.zeros(backend.ka, T, 0, 0, 0)

# Reduced-interface transfer accounting: the per-apply device/host
# copies of the interface values and correction factors, tallied apart from
# the halo staging so the two shares are readable separately. Active only
# under `TRACK_DEVICE_TRANSFERS` (halo.jl), like the halo counters.
const REDUCED_TRANSFER_BYTES = Ref(0)
const REDUCED_TRANSFER_TIME = Ref(0.0)

@inline function _reduced_copy!(dest, src)
    if TRACK_DEVICE_TRANSFERS[]
        t0 = time_ns()
        copyto!(dest, src)
        REDUCED_TRANSFER_TIME[] += (time_ns() - t0) / 1e9
        REDUCED_TRANSFER_BYTES[] += length(src) * sizeof(eltype(src))
    else
        copyto!(dest, src)
    end
    return dest
end

"""
    backend_plan(backend, plan)

Construction-time plan routing: the identity on `CPUBackend`, and
[`device_plan`](@ref) on a `DeviceBackend`, so a solver built on device storage
stores `DevicePlan`s and every `apply_along!` below the step drivers runs the
device kernels without a per-call conversion. Every plan-construction site of
the `Solver` constructor goes through here, fold parity plans included.
"""
backend_plan(::CPUBackend, plan::AbstractDirPlan) = plan
backend_plan(backend::DeviceBackend, plan::AbstractDirPlan) =
    device_plan(plan, backend.ka)

# Device mirrors of the Thomas factorization and its spike vectors, plus the
# interface staging buffers: `ends` packs (y₁, yₙ) per line in the layout of
# `line_solver.ends`, and `zp`/`zn` carry the per-line correction values back
# from the host reduced solve.
struct DeviceThomas{VT,MT}
    l::VT
    dinv::VT
    c::VT
    v::VT
    w::VT
    ends::MT
    zp::VT
    zn::VT
end

# Device mirrors of the banded factorization: `L` (q × n) multipliers, `U`
# ((q+1) × n) upper bands, `V`/`W` (n × q) spike blocks, `ends` (2q × lines),
# and `zbp`/`zbn` (lines × q) correction values.
struct DeviceBanded{MT}
    q::Int
    L::MT
    U::MT
    V::MT
    W::MT
    ends::MT
    zbp::MT
    zbn::MT
end

"""
    DevicePlan

Device execution plan for a compact operator: the coefficient and
factorization mirrors of a host [`DirPlan`](@ref) or [`BandPlan`](@ref) on a
KernelAbstractions backend, with a (lines × n) packed-line buffer. Built by
[`device_plan`](@ref); applied through the same [`apply_along!`](@ref) entry
point as the host plans, on fields living on the same backend. The reduced
interface stage runs on the host through the wrapped plan's `line_solver`, so
its collectives keep the ordering of the host path and every rank of the
sub-communicator must call `apply_along!`.
"""
struct DevicePlan{T,P<:AbstractDirPlan,KB,S,MB<:AbstractMatrix{T},
                  VT<:AbstractVector{T},MC<:AbstractMatrix{T},
                  VI<:AbstractVector{Int}} <: AbstractDirPlan
    host::P            # wrapped plan: metadata, line_solver, reduced-stage workspaces
    backend::KB
    dim::Int
    n::Int
    lines::Int
    sym::Bool
    lo_closed::Bool
    hi_closed::Bool
    colwise::Bool      # dim == 1: mirror the solve_col! banded arithmetic
    a0::T
    nclo::Int
    nchi::Int
    B::MB              # (lines × n) packed-line buffer
    ci::VT
    clo::MC            # closure stencils padded to a rectangle, one column per row
    clo_first::VI
    clo_len::VI
    chi::MC
    chi_first::VI
    chi_len::VI
    sweep::S
end

# Pad the ragged closure stencils to a dense (width × rows) rectangle; the
# fill kernel loops each column to its stored length, so the zero padding is
# never read.
function _pad_stencils(rows::Vector{Vector{T}}) where {T}
    width = max(isempty(rows) ? 0 : maximum(length, rows), 1)
    pad = zeros(T, width, length(rows))
    for (j, r) in enumerate(rows), κ in eachindex(r)
        pad[κ, j] = r[κ]
    end
    return pad, Int[length(r) for r in rows]
end

"""
    device_plan(plan, backend) -> DevicePlan

Mirror a host [`DirPlan`](@ref) or [`BandPlan`](@ref) onto a
KernelAbstractions `backend`. Coefficients, factorizations, and spikes are
adapted to the backend once here; the packed-line and interface staging
buffers are allocated on it. With the KernelAbstractions CPU backend the
mirrors alias the host arrays, which is how the test suite compares the two
paths bitwise on one machine.
"""
function device_plan(plan::DirPlan{T}, backend) where {T}
    ls = plan.line_solver
    sweep = DeviceThomas(adapt(backend, ls.F.l), adapt(backend, ls.F.dinv),
                         adapt(backend, ls.F.c), adapt(backend, ls.v),
                         adapt(backend, ls.w),
                         KernelAbstractions.zeros(backend, T, 2, plan.lines),
                         KernelAbstractions.zeros(backend, T, plan.lines),
                         KernelAbstractions.zeros(backend, T, plan.lines))
    return _device_plan(plan, backend, sweep)
end

function device_plan(plan::BandPlan{T}, backend) where {T}
    ls = plan.line_solver
    q = ls.q
    sweep = DeviceBanded(q, adapt(backend, ls.F.L), adapt(backend, ls.F.U),
                         adapt(backend, ls.V), adapt(backend, ls.W),
                         KernelAbstractions.zeros(backend, T, 2q, plan.lines),
                         KernelAbstractions.zeros(backend, T, plan.lines, q),
                         KernelAbstractions.zeros(backend, T, plan.lines, q))
    return _device_plan(plan, backend, sweep)
end

function _device_plan(plan::AbstractDirPlan, backend, sweep)
    T = eltype(plan.ci)
    clo_pad, clo_len = _pad_stencils(plan.clo)
    chi_pad, chi_len = _pad_stencils(plan.chi)
    return DevicePlan(plan, backend, plan.dim, plan.n, plan.lines,
                      plan.scheme.symmetric, plan.lo_closed, plan.hi_closed,
                      plan.dim == 1, plan.a0, length(plan.clo), length(plan.chi),
                      KernelAbstractions.zeros(backend, T, plan.lines, plan.n),
                      adapt(backend, plan.ci),
                      adapt(backend, clo_pad), adapt(backend, plan.clo_first),
                      adapt(backend, clo_len),
                      adapt(backend, chi_pad), adapt(backend, plan.chi_first),
                      adapt(backend, chi_len),
                      sweep)
end

# --- Kernels ----------------------------------------------------------------
# The fill and scatter iterate (a, b, i): the two orthogonal coordinates in
# ascending dimension order, then the sweep coordinate, so the line index is
# l = (b−1)·n_o1 + a, the same line ordering as both host layouts. `a` is
# the first ndrange dimension, so for the y and z sweeps consecutive threads
# run along x: field reads and B writes coalesce.

@kernel function _dev_fill_kernel!(B, @Const(f), @Const(ci), @Const(clo),
                                   @Const(clo_first), @Const(clo_len),
                                   @Const(chi), @Const(chi_first),
                                   @Const(chi_len), n, nclo, nchi, a0, sym,
                                   lo_closed, hi_closed, n_o1, pad,
                                   ::Val{D}) where {D}
    a, b, i = @index(Global, NTuple)
    @inbounds begin
        l = (b - 1) * n_o1 + a
        if lo_closed && i <= nclo
            i0 = clo_first[i] - 1
            acc = zero(eltype(B))
            for κ in 1:clo_len[i]
                acc += clo[κ, i] * f[_gidx(Val(D), i0 + κ, a, b, pad)]
            end
            B[l, i] = acc
        elseif hi_closed && i > n - nchi
            jr = n + 1 - i
            i0 = n + 2 - chi_first[jr]
            acc = zero(eltype(B))
            for κ in 1:chi_len[jr]
                acc += chi[κ, jr] * f[_gidx(Val(D), i0 - κ, a, b, pad)]
            end
            B[l, i] = acc
        elseif sym
            acc = a0 * f[_gidx(Val(D), i, a, b, pad)]
            for m in eachindex(ci)
                acc += ci[m] * (f[_gidx(Val(D), i + m, a, b, pad)] +
                                f[_gidx(Val(D), i - m, a, b, pad)])
            end
            B[l, i] = acc
        else
            acc = zero(eltype(B))
            for m in eachindex(ci)
                acc += ci[m] * (f[_gidx(Val(D), i + m, a, b, pad)] -
                                f[_gidx(Val(D), i - m, a, b, pad)])
            end
            B[l, i] = acc
        end
    end
end

@kernel function _dev_scatter_kernel!(out, @Const(B), n_o1, pad, ::Val{D}) where {D}
    a, b, i = @index(Global, NTuple)
    @inbounds out[_gidx(Val(D), i, a, b, pad)] = B[(b - 1) * n_o1 + a, i]
end

# One thread per line, the sweep loop inside the thread. Consecutive threads
# read and write consecutive rows of B at every sweep position.
@kernel function _dev_thomas_kernel!(B, @Const(lmul), @Const(dinv), @Const(c), n)
    l = @index(Global, Linear)
    @inbounds begin
        for i in 2:n
            B[l, i] -= lmul[i] * B[l, i-1]
        end
        B[l, n] *= dinv[n]
        for i in (n-1):-1:1
            B[l, i] = (B[l, i] - c[i] * B[l, i+1]) * dinv[i]
        end
    end
end

@kernel function _dev_banded_kernel!(B, @Const(Lm), @Const(U), n, q)
    l = @index(Global, Linear)
    @inbounds begin
        for k in 1:(n-1)
            xk = B[l, k]
            for m in 1:min(q, n - k)
                lm = Lm[m, k]
                # Skip a zero multiplier, as all three host sweeps do.
                # Identical to multiplying on finite data, but 0 * Inf is
                # NaN, so the skip keeps a non-finite field from
                # spreading along rows the host leaves untouched. One
                # comparison per line-step.
                iszero(lm) && continue
                B[l, k+m] -= lm * xk
            end
        end
        for i in n:-1:1
            acc = B[l, i]
            for t in 1:min(q, n - i)
                acc -= U[1+t, i] * B[l, i+t]
            end
            # Reciprocal pivot on every dimension, as all three host banded
            # back-substitutions now do (`solve_col!` carries the reason).
            B[l, i] = acc * inv(U[1, i])
        end
    end
end

@kernel function _dev_pack_ends2_kernel!(ends, @Const(B), n)
    l = @index(Global, Linear)
    @inbounds begin
        ends[1, l] = B[l, 1]
        ends[2, l] = B[l, n]
    end
end

@kernel function _dev_pack_endsq_kernel!(ends, @Const(B), n, q)
    l = @index(Global, Linear)
    @inbounds for r in 1:q
        ends[r, l] = B[l, r]
        ends[q+r, l] = B[l, n-q+r]
    end
end

@kernel function _dev_correct2_kernel!(B, @Const(v), @Const(w), @Const(zp),
                                       @Const(zn))
    l, i = @index(Global, NTuple)
    @inbounds B[l, i] -= v[i] * zp[l] + w[i] * zn[l]
end

@kernel function _dev_correctq_kernel!(B, @Const(V), @Const(W), @Const(zbp),
                                       @Const(zbn), q, colwise)
    l, i = @index(Global, NTuple)
    @inbounds if colwise
        acc = zero(eltype(B))
        for t in 1:q
            acc += V[i, t] * zbp[l, t] + W[i, t] * zbn[l, t]
        end
        B[l, i] -= acc
    else
        for t in 1:q
            B[l, i] -= V[i, t] * zbp[l, t] + W[i, t] * zbn[l, t]
        end
    end
end

# --- Application ------------------------------------------------------------

function _dev_fill!(plan::DevicePlan, f, decomp::Decomp, ::Val{D}) where {D}
    o1, o2 = _odims(Val(D))
    n_o1 = decomp.n_local[o1]
    n_o2 = decomp.n_local[o2]
    _dev_fill_kernel!(plan.backend)(
        plan.B, f, plan.ci, plan.clo, plan.clo_first, plan.clo_len,
        plan.chi, plan.chi_first, plan.chi_len, plan.n, plan.nclo, plan.nchi,
        plan.a0, plan.sym, plan.lo_closed, plan.hi_closed, n_o1,
        decomp.n_halo_d, Val(D); ndrange=(n_o1, n_o2, plan.n))
    _maybe_sync(plan.backend)
    return nothing
end

function _dev_scatter!(out, plan::DevicePlan, decomp::Decomp, ::Val{D}) where {D}
    o1, o2 = _odims(Val(D))
    n_o1 = decomp.n_local[o1]
    n_o2 = decomp.n_local[o2]
    _dev_scatter_kernel!(plan.backend)(
        out, plan.B, n_o1, decomp.n_halo_d, Val(D);
        ndrange=(n_o1, n_o2, plan.n))
    _maybe_sync(plan.backend)
    return nothing
end

function _dev_solve!(plan::DevicePlan, sweep::DeviceThomas)
    ls = plan.host.line_solver
    ls.explicit && return nothing   # identity LHS: the fill is the answer
    L, n = plan.lines, plan.n
    _dev_thomas_kernel!(plan.backend)(plan.B, sweep.l, sweep.dinv, sweep.c, n;
                                      ndrange=L)
    _maybe_sync(plan.backend)
    ls.hasred || return nothing

    _dev_pack_ends2_kernel!(plan.backend)(sweep.ends, plan.B, n; ndrange=L)
    # The one unconditional fence per apply: the host is about to read
    # the packed interface values, so the queue must have drained to here.
    # The H2D copies below block the host until complete, and every later
    # kernel queues after them on the same in-order stream, so no further
    # fence is needed on the way back down.
    synchronize(plan.backend)
    _reduced_copy!(ls.ends, sweep.ends)
    _reduced_solve!(ls, L)
    _reduced_copy!(sweep.zp, ls.zbp)
    _reduced_copy!(sweep.zn, ls.zbn)
    _dev_correct2_kernel!(plan.backend)(plan.B, sweep.v, sweep.w, sweep.zp,
                                        sweep.zn; ndrange=(L, n))
    _maybe_sync(plan.backend)
    return nothing
end

function _dev_solve!(plan::DevicePlan, sweep::DeviceBanded)
    ls = plan.host.line_solver
    L, n, q = plan.lines, plan.n, sweep.q
    # No `colwise`: the elimination sweep and the pivot are now the same
    # arithmetic on every dimension. Only the spike correction below still
    # follows the plan's dimension.
    _dev_banded_kernel!(plan.backend)(plan.B, sweep.L, sweep.U, n, q;
                                      ndrange=L)
    _maybe_sync(plan.backend)
    ls.hasred || return nothing

    _dev_pack_endsq_kernel!(plan.backend)(sweep.ends, plan.B, n, q; ndrange=L)
    synchronize(plan.backend)   # the reduced-solve fence; see the tridiagonal method
    _reduced_copy!(ls.ends, sweep.ends)
    _reduced_solve!(ls, L)
    _reduced_copy!(sweep.zbp, ls.zbp)
    _reduced_copy!(sweep.zbn, ls.zbn)
    _dev_correctq_kernel!(plan.backend)(plan.B, sweep.V, sweep.W, sweep.zbp,
                                        sweep.zbn, q, plan.colwise;
                                        ndrange=(L, n))
    _maybe_sync(plan.backend)
    return nothing
end

"""
    apply_along!(out, plan::DevicePlan, f, decomp)

The device counterpart of the host method: fill, sweep, spike correction, and
scatter run as kernels on `plan.backend`, with the reduced interface stage on
the host. `out` and `f` must live on the same backend as the plan, and `f`
must have current rank-boundary halos. Like the host method, it is collective:
every rank of the sub-communicator along `plan.dim` must call this.
"""
function apply_along!(out, plan::DevicePlan, f, decomp::Decomp)
    d = plan.dim
    if d == 1
        _dev_fill!(plan, f, decomp, Val(1))
        _dev_solve!(plan, plan.sweep)
        _dev_scatter!(out, plan, decomp, Val(1))
    elseif d == 2
        _dev_fill!(plan, f, decomp, Val(2))
        _dev_solve!(plan, plan.sweep)
        _dev_scatter!(out, plan, decomp, Val(2))
    else
        _dev_fill!(plan, f, decomp, Val(3))
        _dev_solve!(plan, plan.sweep)
        _dev_scatter!(out, plan, decomp, Val(3))
    end
    return out
end

# The fused-scatter entry points are host-only; a device solver routes through
# the two-pass form (apply into scratch, then a pointwise device kernel) in
# `deriv_scaled_along!` / `div_subtract_along!`. Reaching these methods
# implies that routing broke, so they raise an error before any scalar
# indexing of device arrays.
apply_along_scaled!(out, plan::DevicePlan, f, decomp::Decomp, scale) =
    error("apply_along_scaled! is host-only; a DevicePlan takes the two-pass " *
          "route in deriv_scaled_along!")
apply_along_subtract!(dQ, c::Int, plan::DevicePlan, f, decomp::Decomp, inv_J) =
    error("apply_along_subtract! is host-only; a DevicePlan takes the " *
          "two-pass route in div_subtract_along!")
