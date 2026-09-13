# Julia 1.12.7 codegen crash under multithreaded test run (Windows)

**Status:** diagnosis for a JuliaLang/julia bug report. Not a CompactLES logic
bug: the `device line solves` testset asserts *bitwise* host==device equality,
so it is the first check to trip when codegen misbehaves.

## Symptom

Running the CompactLES test suite with `JULIA_NUM_THREADS=8` (the environment
default here) crashes nondeterministically. Across repeated runs the outcome is
one of: full pass (`device line solves ... 40 40`), a hard crash before or at
some testset, or (as originally reported) `device line solves` failing 4/40 on a
bitwise `==`. **Three** distinct `EXCEPTION_ACCESS_VIOLATION` signatures have been
seen, all in the compile path:

```
best_tbaa           at C:/workdir/src/codegen.cpp:1612   (TBAA metadata)
typekeyvalue_hash   at C:/workdir/src/jltypes.c:1828     (type interning)
llvm::MachineInstr::setPCSections ... in libLLVM-18jl.dll (SelectionDAG emit)
```

The **crash source line moves between runs**: one run died compiling
the top-level expression at `test/runtests.jl:179` (a `Float32` static-AMR
testset), another at the `device line solves` testset (`:2966`). A fault whose
location floats across unrelated top-level expressions, with three different
signatures all inside LLVM/type-intern code, is the signature of a
**concurrent-JIT race on shared compiler state**, not a bug in any one method.

### Full backtrace of the `setPCSections` crash (most informative)

The access violation is inside LLVM SelectionDAG instruction emission, reached
from Julia's on-demand JIT (`jl_compile_codeinst_now` → `SimpleCompiler` →
legacy `PassManager` → `X86DAGToDAGISel`) while first-compiling a top-level
method during `jl_interpret_toplevel_thunk`:

```
_ZN4llvm12MachineInstr13setPCSectionsERNS_15MachineFunctionEPNS_6MDNodeE   (libLLVM-18jl.dll)
_ZN4llvm7BuildMIE...                                                        (libLLVM-18jl.dll)
_ZN4llvm12InstrEmitter15EmitSpecialNodeE...                                 (libLLVM-18jl.dll)
_ZN4llvm18ScheduleDAGSDNodes12EmitScheduleE...                              (libLLVM-18jl.dll)
_ZN4llvm16SelectionDAGISel17CodeGenAndEmitDAGEv                             (libLLVM-18jl.dll)
_ZN4llvm16SelectionDAGISel20SelectAllBasicBlocksE...                        (libLLVM-18jl.dll)
_ZN12_GLOBAL__N_115X86DAGToDAGISel20runOnMachineFunctionE...                (libLLVM-18jl.dll)
llvm::legacy::PassManagerImpl::run(llvm::Module&)                           (libLLVM-18jl.dll)
llvm::orc::SimpleCompiler::operator()(llvm::Module&)                        (libLLVM-18jl.dll)
operator()          at jitlayers.cpp:1561
addModule           at jitlayers.cpp:2056
jl_compile_codeinst_now   at jitlayers.cpp:626
jl_compile_codeinst_impl  at jitlayers.cpp:824
jl_compile_method_internal at gf.c:3528
ijl_apply_generic   at gf.c:4214
...
jl_interpret_toplevel_thunk at interpreter.c:898
```

All three signatures sit in the compile path (TBAA metadata; type interning;
machine-instruction PC-section metadata), consistent with a concurrent-JIT /
type-intern race, not a numerics fault.

## Why it is not a CompactLES bug

The host and device compact-line-solve paths are identical operation-for-
operation (fill, tridiagonal/banded solve, spike correction; verified by reading
`operators.jl`, `tridiag.jl`, `banded.jl`, `lines_transposed.jl`,
`lines_device.jl`). Empirically, in a **fresh single-purpose process** the two
paths agree **bitwise across 36,000 comparisons** (full scheme × dim × periodic ×
fold matrix, 1000× each, seed 7). No divergence at 1/4/8/32 threads in isolation.
The mismatch and the crashes only appear inside the full multithreaded suite.

## Environment

```
Julia 1.12.7 (2026-08-15) official release
OS: Windows x86_64-w64-mingw32
CPU: 32 × Intel i9-14900K (Alder/Raptor Lake hybrid P+E cores)
LLVM: libLLVM-18.1.7 (ORCJIT, alderlake)
Threads: 8 (JULIA_NUM_THREADS=8), 8 GC threads
KernelAbstractions 0.9.42, MPI 0.20.26, StaticArrays 1.9.19, Adapt 4.7.0
```

CI (Linux, same 1.12.7) passes. Local Windows fails. Points to a Windows/
threaded-ORCJIT-specific fault.

## Reproduction recipe

1. `cd CompactLES` at commit `0f04783` (clean tree).
2. Ensure `JULIA_NUM_THREADS=8` (or `-t 8`).
3. Run `julia --project=. test/runtests.jl` ~5×; expect a hard crash on
   roughly a third of runs (or, more rarely, `device line solves` 4/40).
4. Contrast: `julia --project=. -t 1 test/runtests.jl`.

## Thread-count contrast (measured)

| threads | runs | crashes | `device line solves` |
|---|---|---|---|
| `-t 1` | 2 | 0 | 40/40, 40/40 |
| `-t 8` | (observed) | ~1 in 3 | crash, or 4/40, or 40/40 |

`-t 1` is stable: two consecutive full-suite runs completed with exit 0 and the
bitwise device testset at 40/40 (2.4 s / 2.5 s). Every crash and the 4/40 miss
have only ever been seen at `-t 8`. This is the core thread-count correlation.

## Narrowing still open

- **A standalone, CompactLES-free reproducer has not yet triggered the crash.**
  Three synthetic stressors were tried at `-t 8`: (a) 400 distinct
  arithmetic-loop functions, (b) struct-of-arrays + tuple functions, (c)
  tuple/`ntuple` + mutable-struct + small-Union functions, each generated into a
  fresh `Module` per rep and first-compiled concurrently via
  `Threads.@threads :static` + `Base.invokelatest`, 250–400 functions × 15–25
  reps. All completed cleanly. The earlier observation stands: the trigger is
  concurrent **first** compilation, but synthetic method bodies do not reproduce
  it; the real suite's mix of deeply parametric specializations (parametric on
  `T`/`Metric`/`EOS`/`BoundaryCondition`, StaticArrays, function barriers)
  appears necessary. **For now the minimal reproducer is the package suite at
  `-t 8`.** (Scratch scripts `tmp_jit_race*.jl` were used for this and removed.)
- Next reduction step: bisect which `include(...)` / testset block in
  `test/runtests.jl` first crashes at `-t 8`. This is statistically noisy (the
  fault floats and fires ~1/3 of runs, each run minutes long), so it needs many
  trials per candidate.
- Workarounds to test upstream: `--compile=min`, and any knob that serializes
  JIT compilation across threads.
