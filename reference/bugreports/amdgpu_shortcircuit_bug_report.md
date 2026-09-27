# LLVM 22.1.8 StructurizeCFG drops a phi of two uniform values on AMDGPU

**Status:** diagnosis for an LLVM bug report, with the AMDGPU.jl exposure.
Found 2026-09-26 when `bench/device_solver.jl`'s resolved-θ axis case took
8 device steps against 7 on the CPU, on the workstation's RX 6800 XT
(gfx1030). Worked around in CompactLES commit 1345d40. Not yet filed
upstream.

## The fault

A phi whose incoming values are two uniform values (kernel arguments), at
the join of a branch whose condition varies per thread, loses one of its
incoming values in the `structurizecfg` pass of the LLVM 22.1.8 AMDGPU code
generator. Every thread then receives the other value. In Julia the shape is
a short-circuit condition in a ternary:

```julia
using AMDGPU

function k3!(out, s, t, h, a, b)
    i = workitemIdx().x
    v = t == 1 ? i : t == 2 ? 2i : 3i
    @inbounds out[i] = (s == 0 || v <= h) ? a : b
    return nothing
end

o = ROCArray(zeros(Int, 32))
@roc groupsize=32 k3!(o, 1, 1, 8, 1, -1)
Array(o)[1:12]   # expected eight 1s, then -1; every element is -1
```

Whether a given source reaches the faulty shape depends on how Julia lowers
it. The form CompactLES hit (`v = sd == 1 ? i : j`, the same condition, a
2-D array) fails under Julia 1.11 only; Julia 1.12 and 1.13 lower it to a
`switch` whose structurized form is correct. `k3!` fails on all three.
AMDGPU.jl 2.8.0, GPUCompiler 2.6.0, HIP 6.4.50101, RX 6800 XT:

| Julia (in-process LLVM) | CompactLES form | `k3!` | `k3!`, in-process code generation |
|---|---|---|---|
| 1.11.4 (16.0.6) | wrong | wrong | correct |
| 1.12.7 (18.1.7) | correct | wrong | correct |
| 1.13.0 (20.1.8) | correct | wrong | correct |

The first report was on AMDGPU.jl 2.7.2 with GPUCompiler 2.2.1 and
`AMDGPU_LLVM_Backend_jll` 22.1.8+1; the table uses 22.1.8+2. `ifelse(c1 | c2,
a, b)` in place of the ternary is correct for the CompactLES form on every
version. The HIP SDK's
ROCm 7.1 directory on this machine is empty, so only ROCm 6.4 was
available; ROCm supplies the runtime and `ld.lld` here, not the code
generator.

## The layer

GPUCompiler 2.6 does not generate GCN with the Julia process's own libLLVM.
When `AMDGPU_LLVM_Backend_jll` is available, which it is in every AMDGPU.jl
environment, `GCNCompilerTarget` defaults to `backend = :external`, and
`mcgen` writes the optimized module as bitcode and runs that package's
`llc` (LLVM 22.1.8) on it. The Julia version therefore changes only the IR
handed to `llc`. Overriding `mcgen` to take the in-process path makes every
failing row correct (last column above), so upstream LLVM 16.0.6, 18.1.7 and
20.1.8 all compile the kernel correctly and the fault entered in 21 or 22.

The IR is correct on entry to `llc`, and the fault reproduces with `llc`
alone:

```llvm
target triple = "amdgcn-amd-amdhsa"

; out[tid] = (s == 0 || (t ? tid <= h : tid <=u h)) ? a : b; s, t, h, a, b uniform
define amdgpu_kernel void @k(ptr addrspace(1) %out, i32 %s, i1 %t, i32 %h, i64 %a, i64 %b) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %s0 = icmp eq i32 %s, 0
  br i1 %s0, label %take_a, label %pick
pick:
  br i1 %t, label %signed, label %unsigned
signed:
  %c1 = icmp sle i32 %tid, %h
  br label %merge
unsigned:
  %c2 = icmp ule i32 %tid, %h
  br label %merge
merge:
  %c = phi i1 [ %c1, %signed ], [ %c2, %unsigned ]
  br i1 %c, label %take_a, label %join
take_a:
  br label %join
join:
  %x = phi i64 [ %a, %take_a ], [ %b, %merge ]
  %p = getelementptr inbounds i64, ptr addrspace(1) %out, i32 %tid
  store i64 %x, ptr addrspace(1) %p
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
```

`llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030`:

| `llc` | result |
|---|---|
| 16.0.6jl (`LLVM_jll`, Julia 1.11's) | correct |
| AMD LLVM 20.0.0git (HIP SDK 6.4, AOMP-18.0-12) | correct |
| 22.1.8 (`AMDGPU_LLVM_Backend_jll` 22.1.8+2) | wrong for gfx1030, gfx1100, gfx908, gfx90a and gfx942 |

`-print-after-all` places the fault in `structurizecfg`. Before it, `join`
holds `%x = phi i64 [ %a.load2, %take_a ], [ %b.load3, %merge ]`, with
`%a.load2` (an `extractelement` from the kernel-argument load) moved into
`take_a` by the `sink` pass. After it:

```llvm
Flow6:                                  ; preds = %merge, %entry
  %6 = phi i64 [ %b.load3, %merge ], [ %a.load2, %entry ]
  %7 = phi i1 [ %c, %merge ], [ true, %entry ]
  br i1 %7, label %take_a, label %join
take_a:                                 ; preds = %Flow6
  br label %join
join:                                   ; preds = %take_a, %Flow6
  ...
  store i64 %6, ptr addrspace(1) %p, align 8
```

`%a.load2` has been hoisted into `entry` and the phi in `join` replaced by
`%6`, which is `%b` on every path through `merge`. The path `merge →
take_a → join` (the per-thread condition true) should yield `%a` and yields
`%b`. In the GCN the masked region that should copy `a` into the result is
empty:

```
; llc 22.1.8                          ; llc 16.0.6 and AMD 20.0.0git
.LBB0_6:  ; %Flow6                    .LBB0_6:  ; %Flow6
  s_and_saveexec_b32 s1, s0             v_mov_b32_e32 v1, s6      ; b
  s_or_b32 exec_lo, exec_lo, s1         v_mov_b32_e32 v2, s7
  ...                                   s_and_saveexec_b32 s1, s0
  v_mov_b32_e32 v0, s6    ; b           v_mov_b32_e32 v1, s4      ; a, masked
  v_mov_b32_e32 v1, s7                  v_mov_b32_e32 v2, s5
                                        s_or_b32 exec_lo, exec_lo, s1
```

The hoisting of a zero-cost incoming value out of the "else" block matches
`hoistZeroCostElseBlockPhiValues` in `StructurizeCFG.cpp`, added in 2025
(committed, reverted in July 2025, and relanded). The attribution is
unconfirmed; no LLVM 21 build was at hand to narrow the range.

## How it reached the package

`_delta4_signed_point!` (`src/artificial.jl`) chose the mirror sign of the
paired-fold detector as `(sd == 0 || v <= half) ? sgn_a : sgn_b`. On the
GPU, under Julia 1.11, every thread took `sgn_b`, so the resolved-θ axis
sensed the lower half of θ with the wrong sign. The KernelAbstractions CPU
backend never reaches this code generator, so the serial and MPI suites
cannot see it.

## What was done

Commit 1345d40 writes the choice as `ifelse((sd == 0) | (v <= half),
sgn_a, sgn_b)`, which Julia emits as a `select` with no control flow to
structurize. The device battery of `bench/device_solver.jl` is bitwise
against the CPU again. A search of `src/` for a short-circuit condition
written on the same line as a ternary found no other in a device body; the
CLAUDE.md trap keeps new ones out. Literal constants as the two values
compile correctly (they are not instructions and cannot be hoisted); kernel
arguments do not. Every AMD target,
gfx942 (MI300A) included, compiles through the same `llc`, so the rule holds
on the cluster as well.
