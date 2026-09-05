# Device bring-up check for the pointwise kernels: launch the
# plain-argument bodies on a GPU through `pointwise_ka!` and through
# `pointwise!`'s automatic routing, compare against the CPU bitwise, and time
# a stencil-shaped and a launch-bound body. First measured on the reference
# workstation's RX 6800 XT (AMDGPU.jl v2.7.2 on Windows against the HIP SDK);
# the recorded numbers are in `reference/AMR_GPU.md` (pointwise kernels).
#
# Device packages are not CompactLES dependencies, so this script must run
# from an environment carrying CompactLES and the device package:
#
#   julia --project=<env-with-AMDGPU> -t 8 bench/device_bringup.jl backend=amdgpu
#   julia --project=<env-with-CUDA>   -t 8 bench/device_bringup.jl backend=cuda
#
# Do not pass a bare `Vector{<:AbstractArray}` or `Matrix{<:AbstractArray}`
# kernel argument to a device launch: it does not raise an error, it hangs
# in kernel-argument adaptation (measured on AMDGPU). The FieldVector /
# FieldMatrix wrappers (src/pointwise.jl) are the supported route — zero-cost
# on the host, adapted to isbits tuples at launch — and the section below
# runs the full flux-assembly body through them, Adapt-mirrored EOS included.

using CompactLES
import CompactLES: Decomp, device_plan
using Printf
const CL = CompactLES
using KernelAbstractions
const KA = KernelAbstractions

opt = CompactLES.script_args(ARGS, (backend = "amdgpu", n = 64, reps = 100))

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

    # δ⁴ detector body, clamped edges, all three directions. The sensor weight
    # is the local physical spacing h/inv_h, so the body takes the geometry
    # array; a flat one stands in for an unstretched Cartesian grid.
    f0 = randn(dims[1:3]...)
    outc = zeros(dims[1:3]...)
    ih1 = ones(dims[1:3]...)
    n3p = dims[3]
    fg = device_array(f0)
    ihg1 = device_array(ih1)
    outg = device_array(zeros(dims[1:3]...))
    for dd in 1:3
        CL.pointwise_ka!(CL._delta4_point!, KA.CPU(), n..., outc, f0, 1.0, ih1,
                         1, dd, n[dd], 1, n[dd], false, false, false, n3p,
                         pad, pad, pad)
        CL.pointwise_ka!(CL._delta4_point!, gpu, n..., outg, fg, 1.0, ihg1,
                         1, dd, n[dd], 1, n[dd], false, false, false, n3p,
                         pad, pad, pad)
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
                 outg, fg, 1.0, ihg1, 1, 1, n[1], 1, n[1], false, false, false,
                 n3p, pad, pad, pad))
    t_thr = best(() -> CL.pointwise!(CL._delta4_point!, outc, n...,
                 outc, f0, 1.0, ih1, 1, 1, n[1], 1, n[1], false, false, false,
                 n3p, pad, pad, pad))
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

    # --- Collection-argument bodies through the FieldVector/FieldMatrix
    # wrappers and the Adapt EOS mirror: the full flux assembly, two species.
    nsp = 2
    ncons2 = nsp + 4
    pdims = n .+ 2pad
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)])
    mkp() = abs.(randn(pdims...)) .+ 1.0
    Qc2 = randn(pdims..., ncons2)
    prims = [mkp() for _ in 1:7]                 # rho,u,v,w,p,T_ion,cp_mix
    arts = [abs.(randn(pdims...)) .* 1e-3 for _ in 1:3]
    D_c = [abs.(randn(pdims...)) .* 1e-3 for _ in 1:nsp]
    Y_c = [rand(pdims...) for _ in 1:nsp]
    gu_c = [randn(pdims...) for _ in 1:3, _ in 1:3]
    gT_c = (randn(pdims...), randn(pdims...), randn(pdims...))
    gY_c = [randn(pdims...) for _ in 1:3, _ in 1:nsp]
    fl_c = [zeros(pdims...) for _ in 1:3, _ in 1:ncons2]
    m1, m2, m3, ie = nsp + 1, nsp + 2, nsp + 3, nsp + 4
    fluxargs(Q, pr, ar, D, Y, gu, gT, gY, fl) =
        (Q, eos, pr[1], pr[2], pr[3], pr[4], pr[5], pr[6], pr[7],
         ar[1], ar[2], ar[3], CL.FieldVector(D), CL.FieldVector(Y),
         CL.FieldMatrix(gu), gT, CL.FieldMatrix(gY), CL.FieldMatrix(fl),
         1e-3, 0.7, 0.7, nsp, m1, m2, m3, ie, (true, true, true), pad, pad, pad)
    ac = fluxargs(Qc2, prims, arts, D_c, Y_c, gu_c, gT_c, gY_c, fl_c)
    CL.pointwise!(CL._fluxes_point!, Qc2, n..., ac...)
    Qg2 = device_array(Qc2)
    prims_g = map(device_array, prims)
    arts_g = map(device_array, arts)
    D_g = map(device_array, D_c)
    Y_g = map(device_array, Y_c)
    gu_g = [device_array(gu_c[a, b]) for a in 1:3, b in 1:3]
    gT_g = map(device_array, gT_c)
    gY_g = [device_array(gY_c[dd, s]) for dd in 1:3, s in 1:nsp]
    fl_g = [device_array(zeros(pdims...)) for _ in 1:3, _ in 1:ncons2]
    ag = fluxargs(Qg2, prims_g, arts_g, D_g, Y_g, gu_g, gT_g, gY_g, fl_g)
    CL.pointwise_ka!(CL._fluxes_point!, gpu, n..., ag...)
    d = maximum(maximum(abs.(Array(fl_g[i]) .- fl_c[i])) for i in eachindex(fl_c))
    println("flux assembly  max |gpu - cpu| = ", d, d == 0 ? "  (bitwise)" : "")
    t_gpu = best(() -> CL.pointwise_ka!(CL._fluxes_point!, gpu, n..., ag...))
    t_thr = best(() -> CL.pointwise!(CL._fluxes_point!, Qc2, n..., ac...))
    @printf("flux assembly %d^3 2sp: device %.3f ms   @threaded(%d) %.3f ms\n",
            opt.n, 1e3t_gpu, Threads.nthreads(), 1e3t_thr)

    # --- Compact line solves through DevicePlan. Fill, sweep, spike
    # correction and scatter run as kernels; the reduced interface stage stays
    # on the host (self-coupling only here — single rank, periodic wrap).
    # Arithmetic mirrors the host path per line, so the comparison is bitwise.
    decomp = Decomp((opt.n, opt.n, opt.n), (true, true, true); dims=(1, 1, 1))
    fh = CL.field(decomp)
    copyto!(fh, randn(size(fh)...))
    CL.exchange_halos!(fh, decomp)
    fg2 = device_array(fh)
    for (label, scheme) in (("C6 ", CL.lele_d1_6(Float64)),
                            ("C10", CL.lele_d1_10(Float64)),
                            ("filt", CL.compact_filter(0.45)))
        for dim in 1:3
            plan = CL.plan_direction(decomp, scheme, dim, 1.0 / opt.n)
            dplan = device_plan(plan, gpu)
            oh = CL.field(decomp)
            og = device_array(CL.field(decomp))
            CL.apply_along!(oh, plan, fh, decomp)
            CL.apply_along!(og, dplan, fg2, decomp)
            dmax = maximum(abs.(view(Array(og), CL.interior(decomp)) .-
                                view(oh, CL.interior(decomp))))
            t_gpu = best(() -> CL.apply_along!(og, dplan, fg2, decomp))
            t_thr = best(() -> CL.apply_along!(oh, plan, fh, decomp))
            @printf("%s line dim %d %d^3: dev %.3f ms  host(%d) %.3f ms  |Δ|=%g%s\n",
                    label, dim, opt.n, 1e3t_gpu, Threads.nthreads(), 1e3t_thr,
                    dmax, dmax == 0 ? "  (bitwise)" : "")
        end
    end

    # Closed edges: the closure-row branches of the fill kernel, C6 and C10.
    decomp_c = Decomp((opt.n, opt.n, opt.n), (false, false, false); dims=(1, 1, 1))
    fc = CL.field(decomp_c)
    copyto!(fc, randn(size(fc)...))
    fcg = device_array(fc)
    for (label, scheme) in (("C6 closed ", CL.lele_d1_6(Float64)),
                            ("C10 closed", CL.lele_d1_10(Float64)))
        for dim in 1:3
            plan = CL.plan_direction(decomp_c, scheme, dim, 1.0 / opt.n)
            dplan = device_plan(plan, gpu)
            oh = CL.field(decomp_c)
            og = device_array(CL.field(decomp_c))
            CL.apply_along!(oh, plan, fc, decomp_c)
            CL.apply_along!(og, dplan, fcg, decomp_c)
            dmax = maximum(abs.(view(Array(og), CL.interior(decomp_c)) .-
                                view(oh, CL.interior(decomp_c))))
            @printf("%s dim %d: max|Δ| = %g%s\n", label, dim, dmax,
                    dmax == 0 ? "  (bitwise)" : "")
        end
    end
    return nothing
end

main(opt, device_array)
