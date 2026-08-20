# Cross-machine device performance survey. One machine's timings are not a
# property of the code: the rzadams MI300A log (bench/logs/rzadams_20260819.txt)
# shows C6 Float64 line solves at 3.0/6.0 ms in dims 1-2 against ~0.3 ms for
# every other scheme x dim, and a 26x Float64/Float32 whole-step ratio on
# hardware whose FP64 vector rate is half its FP32 rate — neither visible on
# the RDNA2 workstation. This script is the instrument for such questions: run
# it unchanged on each machine and compare tables, rather than reading any one
# machine's numbers as a property of the implementation.
#
# Sections:
#   host sanity — a threaded-speedup probe. A core-starved allocation (a
#       flux/srun default of one core per task under several Julia threads)
#       inflates every host-relative ratio 10-30x and is invisible in the
#       timings themselves; this probe makes it visible up front.
#   floors — kernel submission, submission+synchronize round trip, and an
#       idle-queue synchronize. These bound what any launch policy can recover
#       and price the per-apply reduced-solve fence.
#   line solves — scheme (C6/C10/C8 filter) x dim x precision x size, device
#       and host apply walls. A per-apply cost flat across sizes is a
#       latency/serialization floor; one that scales with n is throughput
#       (occupancy, register spill). Methodology matches
#       bench/device_bringup.jl so rows compare with existing logs.
#   TGV steps — warm device s/step at scheme x precision, single species,
#       artificial properties off. If a Float64 step anomaly collapses when
#       C6 is swapped for C10, the line solve owns it; if it persists, it is
#       precision-wide. CPU-vs-device TGV lives in bench/device_solver.jl and
#       is not repeated here.
#
# Device packages are not CompactLES dependencies, so run from an environment
# carrying CompactLES AND the device package:
#
#   julia --project=<env-with-AMDGPU> -t 8 bench/device_floors.jl backend=amdgpu
#   julia --project=<env-with-CUDA>   -t 8 bench/device_floors.jl backend=cuda
#
# First-launch kernel compilation dominates the wall time of a fresh process;
# every printed number is a minimum over reps after warm-up.

using CompactLES
using Printf
const CL = CompactLES
# KernelAbstractions through CompactLES's own import, so the script runs from
# any environment carrying CompactLES plus a device package and nothing else.
const KA = CL.KernelAbstractions

opt = script_args(ARGS, (backend = "amdgpu", sizes = "32,64,96", n = 64,
                         steps = 10, reps = 50))

device_array, ka_backend = if opt.backend == "amdgpu"
    @eval using AMDGPU
    @eval AMDGPU.functional() || error("AMDGPU is not functional here")
    @eval println("device: ", AMDGPU.device())
    @eval (ROCArray, ROCBackend())
elseif opt.backend == "cuda"
    @eval using CUDA
    @eval CUDA.functional() || error("CUDA is not functional here")
    @eval println("device: ", CUDA.device())
    @eval (CuArray, CUDABackend())
else
    error("backend must be amdgpu or cuda, got $(opt.backend)")
end

best(f; reps) = (f(); f(); minimum(@elapsed(f()) for _ in 1:reps))

# A dependent multiply-add chain: it measures fair core time per thread, not
# peak FLOPs, which is exactly what a starved allocation distorts.
function _spin(seed::Float64, iters::Int)
    s = seed
    for _ in 1:iters
        s = muladd(s, 0.999999, 1e-9)
    end
    return s
end

function host_sanity()
    nt = Threads.nthreads()
    @printf("julia %s on %s (%s), %d threads of %d hardware threads\n",
            string(VERSION), Base.Libc.gethostname(), Sys.MACHINE, nt,
            Sys.CPU_THREADS)
    iters = 100_000_000
    sink = zeros(nt)
    _spin(0.5, 1000)
    Threads.@threads :static for t in 1:nt
        sink[t] = _spin(0.5, 1000)
    end
    # Keep every result in `sink`: a discarded _spin call is a pure loop the
    # compiler deletes, and t1 then measures nothing.
    t1 = @elapsed (sink[1] = _spin(0.5, iters))
    tn = @elapsed Threads.@threads :static for t in 1:nt
        sink[t] = _spin(0.5, iters)
    end
    @printf("host threading: %d identical compute-bound tasks take %.2fx one",
            nt, tn / t1)
    println(" task's wall (ideal 1.0)")
    if nt > 1 && tn / t1 > 2.0
        println("  WARNING: threads appear core-starved; host-relative",
                " ratios below are not meaningful.")
        println("  Check the launcher's cores-per-task and binding before",
                " reading any host number.")
    end
    return nothing
end

function floors(gpu, device_array, reps)
    g = device_array(randn(8, 8, 8))
    ih = device_array(ones(8, 8, 8))
    launch() = CL.pointwise_ka!(CL._scale_grad_point!, gpu, 8, 8, 8, g, ih)
    launch()
    KA.synchronize(gpu)
    t_sub = best(launch; reps=reps)
    KA.synchronize(gpu)
    t_rt = best(() -> (launch(); KA.synchronize(gpu)); reps=reps)
    t_idle = best(() -> KA.synchronize(gpu); reps=reps)
    @printf("submission %.1f us   launch+sync %.1f us   idle sync %.1f us\n",
            1e6t_sub, 1e6t_rt, 1e6t_idle)
    return nothing
end

function line_matrix(gpu, device_array, sizes, reps)
    @printf("%-16s", "")
    for nn in sizes
        @printf("  %10s (host)  ", "n=$nn")
    end
    println()
    for T in (Float64, Float32)
        fields = map(sizes) do nn
            decomp = Decomp{T}((nn, nn, nn), (true, true, true); dims=(1, 1, 1))
            fh = CL.field(decomp)
            copyto!(fh, randn(T, size(fh)...))
            CL.exchange_halos!(fh, decomp)
            (decomp=decomp, fh=fh, fg=device_array(fh),
             oh=CL.field(decomp), og=device_array(CL.field(decomp)))
        end
        schemes = (("C6", lele_d1_6(T)), ("C10", lele_d1_10(T)),
                   ("filt", compact_filter(T(0.45), T)))
        for (label, scheme) in schemes, dim in 1:3
            @printf("%s %-5s dim %d ", T === Float64 ? "F64" : "F32",
                    label, dim)
            mismatch = false
            for w in fields
                nn = w.decomp.n_global[1]
                plan = CL.plan_direction(w.decomp, scheme, dim, T(1) / nn)
                dplan = device_plan(plan, gpu)
                CL.apply_along!(w.oh, plan, w.fh, w.decomp)
                CL.apply_along!(w.og, dplan, w.fg, w.decomp)
                dmax = maximum(abs.(view(Array(w.og), CL.interior(w.decomp))
                                    .- view(w.oh, CL.interior(w.decomp))))
                mismatch |= dmax != 0
                t_dev = best(() -> CL.apply_along!(w.og, dplan, w.fg,
                                                   w.decomp); reps=reps)
                t_host = best(() -> CL.apply_along!(w.oh, plan, w.fh,
                                                    w.decomp); reps=reps)
                @printf("  %8.3f (%8.3f)", 1e3t_dev, 1e3t_host)
            end
            println(mismatch ? "   |dev-host| != 0 !" : "")
        end
    end
    return nothing
end

function tgv_build(T, deriv, n, ka_backend)
    per = (PeriodicBC(), PeriodicBC())
    γ = T(1.4)
    c0 = T(10)
    p0 = c0^2 / γ
    s = Solver(n_global=(n, n, n), L_domain=(T(2π), T(2π), T(2π)),
               bcs=(per, per, per), eos=single_species(T),
               transport=Transport{T}(mu0=T(1 / 1600)),
               art=ArtParams{T}(enabled=false),
               deriv=deriv, filt=compact_filter(T(0.45), T),
               cfl=T(0.6), backend=DeviceBackend(ka_backend))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(
        u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), zero(T)),
        p=p0 + one(T) / T(16) * (cos(T(2)x) + cos(T(2)y)) * (cos(T(2)z) + T(2)),
        rho=one(T)))
    return s, Q
end

function tgv_steps(ka_backend, n, steps)
    for T in (Float64, Float32)
        for (label, deriv) in (("C6 ", lele_d1_6(T)), ("C10", lele_d1_10(T)))
            s, Q = tgv_build(T, deriv, n, ka_backend)
            run!(s, Q; tfinal=1e6, nmax=steps)
            s, Q = tgv_build(T, deriv, n, ka_backend)
            run!(s, Q; tfinal=1e6, nmax=steps)
            @printf("TGV %s %s %d^3: device %.4f s/step (warm, %d steps)\n",
                    T === Float64 ? "F64" : "F32", label, n,
                    s.wall_total / s.step, s.step)
        end
    end
    return nothing
end

function main(opt, device_array, ka_backend)
    sizes = parse.(Int, split(opt.sizes, ','))
    println("== host sanity")
    host_sanity()
    println("== launch floors (min over $(opt.reps))")
    floors(ka_backend, device_array, opt.reps)
    println("== line solves, ms/apply (device, host in parens; ",
            "min over $(opt.reps))")
    line_matrix(ka_backend, device_array, sizes, opt.reps)
    println("== TGV device steps")
    tgv_steps(ka_backend, opt.n, opt.steps)
    return nothing
end

main(opt, device_array, ka_backend)
