# Device bring-up check for the G1 pointwise kernels: launch the
# plain-argument bodies on a GPU through `pointwise_ka!` and through
# `pointwise!`'s automatic routing, compare against the CPU bitwise, and time
# a stencil-shaped and a launch-bound body. First measured on the reference
# workstation's RX 6800 XT (AMDGPU.jl v2.7.2 on Windows against the HIP SDK);
# the recorded numbers are in the G1 status block of `reference/AMR_GPU.md`.
#
# Device packages are not CompactLES dependencies, so this script must run
# from an environment carrying CompactLES AND the device package:
#
#   julia --project=<env-with-AMDGPU> -t 8 bench/device_bringup.jl backend=amdgpu
#   julia --project=<env-with-CUDA>   -t 8 bench/device_bringup.jl backend=cuda
#
# Do NOT pass a `Vector{<:AbstractArray}` or `Matrix{<:AbstractArray}` kernel
# argument to a device launch: it does not raise an error, it HANGS in
# kernel-argument adaptation (measured on AMDGPU). The collection-argument
# bodies (fluxes, strain, primitives) stay CPU-side until the Adapt mirrors
# exist.

using CompactLES
using Printf
const CL = CompactLES
using KernelAbstractions
const KA = KernelAbstractions

opt = script_args(ARGS, (backend = "amdgpu", n = 64, reps = 100))

device_array = if opt.backend == "amdgpu"
    @eval using AMDGPU
    @eval AMDGPU.functional() || error("AMDGPU is not functional here")
    @eval println("device: ", AMDGPU.device())
    @eval ROCArray
elseif opt.backend == "cuda"
    @eval using CUDA
    @eval CUDA.functional() || error("CUDA is not functional here")
    @eval println("device: ", CUDA.device())
    @eval CuArray
else
    error("backend must be amdgpu or cuda, got $(opt.backend)")
end

best(f; reps=opt.reps) = (f(); f(); minimum(@elapsed(f()) for _ in 1:reps))

function main(opt, device_array)
    n = (opt.n, opt.n, opt.n)
    pad = 4
    nc = 5
    dims = ((n .+ 2pad)..., nc)
    Q0 = randn(dims...); dQ0 = randn(dims...); du0 = randn(dims...)
    Qc, dQc, duc = copy(Q0), copy(dQ0), copy(du0)
    Qg, dQg, dug = device_array(Q0), device_array(dQ0), device_array(du0)
    gpu = KA.get_backend(Qg)

    # RK update body, per component, both paths.
    for c in 1:nc
        CL.pointwise_ka!(CL._rk_point!, KA.CPU(), n..., Qc, dQc, duc,
                         -0.5, 0.3, 1e-3, c, pad, pad, pad)
        CL.pointwise_ka!(CL._rk_point!, gpu, n..., Qg, dQg, dug,
                         -0.5, 0.3, 1e-3, c, pad, pad, pad)
    end
    d = maximum(abs.(Array(Qg) .- Qc))
    println("rk update      max |gpu - cpu| = ", d, d == 0 ? "  (bitwise)" : "")

    # δ⁴ detector body, clamped edges, all three directions.
    f0 = randn(dims[1:3]...)
    outc = zeros(dims[1:3]...)
    fg = device_array(f0)
    outg = device_array(zeros(dims[1:3]...))
    for dd in 1:3
        CL.pointwise_ka!(CL._delta4_point!, KA.CPU(), n..., outc, f0, 1.0,
                         dd, n[dd], 1, n[dd], false, false, false, pad, pad, pad)
        CL.pointwise_ka!(CL._delta4_point!, gpu, n..., outg, fg, 1.0,
                         dd, n[dd], 1, n[dd], false, false, false, pad, pad, pad)
    end
    d = maximum(abs.(Array(outg) .- outc))
    println("delta4 sensor  max |gpu - cpu| = ", d, d == 0 ? "  (bitwise)" : "")

    # The automatic routing: a device route array sends pointwise! to KA.
    g0 = randn(dims[1:3]...)
    ih0 = 1.0 .+ 0.1 .* rand(dims[1:3]...)
    gg = device_array(g0)
    ihg = device_array(ih0)
    CL.pointwise!(CL._scale_grad_point!, gg, dims[1:3]..., gg, ihg)
    println("routing        max |gpu - cpu| = ",
            maximum(abs.(Array(gg) .- g0 .* ih0)))

    # Timing: one stencil-shaped single launch, and the launch-bound RK
    # update (five launches, each synchronized — the G3 batching case).
    t_gpu = best(() -> CL.pointwise_ka!(CL._delta4_point!, gpu, n...,
                 outg, fg, 1.0, 1, n[1], 1, n[1], false, false, false,
                 pad, pad, pad))
    t_thr = best(() -> CL.pointwise!(CL._delta4_point!, outc, n...,
                 outc, f0, 1.0, 1, n[1], 1, n[1], false, false, false,
                 pad, pad, pad))
    @printf("delta4 (one dir) %d^3:  device %.3f ms   @threaded(%d) %.3f ms\n",
            opt.n, 1e3t_gpu, Threads.nthreads(), 1e3t_thr)
    t_gpu = best(() -> for c in 1:nc
        CL.pointwise_ka!(CL._rk_point!, gpu, n..., Qg, dQg, dug,
                         -0.5, 0.3, 1e-3, c, pad, pad, pad)
    end)
    t_thr = best(() -> for c in 1:nc
        CL.pointwise!(CL._rk_point!, Qc, n..., Qc, dQc, duc,
                      -0.5, 0.3, 1e-3, c, pad, pad, pad)
    end)
    @printf("rk update x%d launches: device %.3f ms   @threaded(%d) %.3f ms\n",
            nc, 1e3t_gpu, Threads.nthreads(), 1e3t_thr)
    return nothing
end

main(opt, device_array)
