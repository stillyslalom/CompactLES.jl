# AMDGPU.jl cannot compile a kernel's exception path under Julia 1.12 and 1.13

**Status:** diagnosis for an AMDGPU.jl (or GPUCompiler.jl) bug report, with
the CompactLES exposure. Found 2026-11-09 when the hardware-GPU check
(`probes/device_bringup.jl`, then `bench/device_solver.jl backend=amdgpu`)
was first run under Julia 1.12 and 1.13 on the workstation's RX 6800 XT
(gfx1030); it last passed under Julia 1.11.4 with AMDGPU.jl 2.7.2. Worked
around in CompactLES for the package's own kernels in the commit that adds
this report. Not yet filed upstream.

## The fault

A kernel in which an exception path survives optimization fails to compile.
The paths tested are the divide-by-zero test of an integer `rem` and the
bounds check of a store outside `@inbounds`; the same kernels without them
compile and run. AMDGPU.jl's own reductions have such a path (the `div` in
`fld1`, `src/kernels/mapreduce.jl`), so `sum(::ROCArray)` fails as well.

```julia
using AMDGPU, KernelAbstractions

@kernel function plain!(out, n)
    i = @index(Global)
    @inbounds out[i] = i + n
end
@kernel function with_rem!(out, n)
    i = @index(Global)
    @inbounds out[i] = rem(i, n)
end
@kernel function unchecked_rem!(out, n)
    i = @index(Global)
    @inbounds out[i] = Core.Intrinsics.srem_int(i, n)
end
@kernel function unbounded_store!(out, n)
    i = @index(Global)
    out[i] = n
end
function roc_rem!(out, n)
    i = workitemIdx().x
    @inbounds out[i] = rem(i, n)
    return nothing
end

function ka(k)
    out = ROCArray(zeros(Int, 8))
    k(ROCBackend(), 8)(out, 3; ndrange=8)
    KernelAbstractions.synchronize(ROCBackend())
    return Array(out)
end

ka(plain!)              # [4, 5, ..., 11]
ka(unchecked_rem!)      # [1, 2, 0, 1, 2, 0, 1, 2]
ka(with_rem!)           # InvalidIRError
ka(unbounded_store!)    # InvalidIRError
out = ROCArray(zeros(Int, 8)); @roc groupsize=8 roc_rem!(out, 3)  # InvalidIRError
sum(ROCArray(Float32[1, 2, 3]))                                  # InvalidIRError

# The same source of AMDGPU.Device.signal_exception, evaluated again:
@eval AMDGPU.Device function signal_exception()
    ei = kernel_state().exception_info
    signal_exception!(ei, ExceptionCode.UNKNOWN)
    return
end
# Every kernel above now compiles and gives the expected values; sum gives 6.0.
```

Each failure carries the same three reasons, from the inlined
`signal_exception` (AMDGPU `src/device/runtime.jl`, lines 76 and 77) under
the throwing operation:

```
Reason: unsupported dynamic function invocation (call to kernel_state() @ AMDGPU.Device none:0)
Reason: unsupported dynamic function invocation (call to getproperty)
Reason: unsupported call to an unknown function (call to jl_f_throw_methoderror)
```

Expected: every kernel compiles, the exception path reporting through
`signal_exception` as it did under Julia 1.11.

| | `plain!` | `unchecked_rem!` | `with_rem!` | `unbounded_store!` | `@roc rem` | `sum` |
|---|---|---|---|---|---|---|
| Julia 1.12.7, AMDGPU 2.7.2, GPUCompiler 2.2.1 | compiles | compiles | fails | fails | fails | fails |
| Julia 1.13.3, AMDGPU 2.8.0, GPUCompiler 2.6.0 | compiles | compiles | fails | fails | fails | fails |
| either, after redefining `signal_exception` | compiles | compiles | compiles | compiles | compiles | compiles |

KernelAbstractions 0.9.45 in both environments, HIP SDK 6.4, Windows 11,
one Julia thread. The rows are one script run in one process per Julia
version: the six kernels as loaded, then the same six after evaluating
`signal_exception` again. No Julia 1.11 environment with AMDGPU.jl was
available, so the passing configuration was not re-run.

## What the tests show

- The fault is not in KernelAbstractions or in CompactLES: a raw `@roc`
  kernel and AMDGPU.jl's own `sum` fail the same way.
- The generator of `@generated kernel_state()` (`runtime.jl`, line 5) works
  in the session: after `signal_exception` alone is evaluated again, with
  `kernel_state` untouched, every kernel compiles, and evaluating
  `kernel_state` again as well changes nothing further.
- The dynamic call therefore comes from compiled code for `signal_exception`
  that exists before the session's first kernel compilation, which the
  redefinition invalidates.

Not established: where that code comes from. Two facts point at the package
image. `GPUCompiler.supports_relocatable_ir()` is `true` in both
environments, and GPUCompiler then persists each runtime-library function's
bitcode with its `CodeInstance` across sessions (`src/rtlib.jl`,
`RuntimeFunctionResults`). AMDGPU.jl's precompile workload compiles a kernel
during precompilation on Julia 1.12 and later only, and its comment
(`src/precompile.jl`) says that `kernel_state()` fails to inline there,
leaving a dynamic call. A runtime library built in that workload would carry
the dynamic call into every later session. Running the reproducer with
`--compiled-modules=no`, under which no package image is built or read and
the workload does not run, would test this; it was not run.

## How it reached the package

Eleven pointwise bodies held an integer `div`, `rem`, `divrem` or `%` of an
extent or a stride: the layer of a stacked launch (`_delta4_point!`,
`_fold_fill_z_point!`, `_edge_fill_z_point!`, `_molecular_flux_point!`,
`_remainder_flux_point!`), the component decode of the Hermite and
interpolation bodies, and the slab decode of the four ring bodies.
`_rate_point!` also read arrays outside `@inbounds` through `_diffusive_rate`
and `_sharpening_rate`. The hardware check stopped on `_rate_point!` and on
`_delta4_point!`; the others were found by reading the bodies. The
KernelAbstractions CPU backend compiles all of them, so the serial and MPI
suites cannot see the fault.

## What was done

The bodies take the quotient and remainder through `_unchecked_div` and
`_unchecked_rem` (`src/pointwise.jl`), the intrinsics `sdiv_int` and
`srem_int` without the divisor test, which give the values of `div` and
`rem` for the positive divisors the bodies divide by, and the two rate
helpers read inside `@inbounds`. The host results are bitwise unchanged.
Under Julia 1.12.7 with AMDGPU.jl 2.7.2 and under 1.13.3 with 2.8.0,
`probes/device_bringup.jl` then compiles every launch and is bitwise against
the CPU. AMDGPU.jl's reductions
keep their exception path, and `max_rate` on device storage reduces with
`maximum`: `bench/device_solver.jl` stops in its first case on
`AMDGPU.partial_mapreduce_device` with the error above. Run through a
wrapper that evaluates `signal_exception` again before including the script,
the whole battery completes, every case bitwise against the CPU. The solver
therefore does not complete a device step under Julia 1.12 or 1.13 until the
fault is fixed upstream or the session evaluates `signal_exception` again.
