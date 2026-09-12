# Julia 1.13.0 miscompiles a recurrence inside `@aliasscope` under `@inbounds`

**Status:** diagnosis for a JuliaLang/julia bug report, with the
KernelAbstractions.jl exposure. Found 2026-09-12 when the `Convergence and
MPI` CI job moved from Julia 1.12.7 to 1.13.0 and the `device line solves`
phase of `test/mpi_tests.jl` failed its bitwise host-vs-device check on the
C6 periodic cases at np = 2 (errors of order 1, not round-off), after which
the `staged device exchange` phase aborted on `sqrt` of a negative number.
Reproduced on the Windows workstation under `julia +1.13`.

## The fault

A loop that carries a dependency through memory (each iteration stores an
element that the next iteration loads) produces wrong values when it sits
inside `Base.Experimental.@aliasscope` and its bounds checks are elided.
No `Base.Experimental.Const` array is involved, so the scope asserts nothing
about the arrays in the loop and the compiler has nothing to rely on.

```julia
using Base.Experimental: @aliasscope

function fwd!(B, a, n)
    for l in axes(B, 1)
        @aliasscope begin
            @inbounds for i in 2:n
                B[l, i] -= a[i] * B[l, i-1]
            end
        end
    end
    return B
end

function fwd_plain!(B, a, n)
    for l in axes(B, 1), i in 2:n
        B[l, i] -= a[i] * B[l, i-1]
    end
    return B
end

B = rand(1, 10); a = rand(10)
maximum(abs.(fwd!(copy(B), a, 10) .- fwd_plain!(copy(B), a, 10)))
```

| Julia | `-O2` (default) | `-O0` | `--check-bounds=yes` |
|---|---|---|---|
| 1.12.7 (LLVM 18.1.7) | 0.0 | 0.0 | 0.0 |
| 1.13.0 (LLVM 20.1.8) | 0.30 (wrong) | 0.0 | 0.0 |
| 1.14.0-DEV.3172 nightly, 2026-09-12 (LLVM 22.1.8) | 0.61 (wrong) | 0.0 | 0.0 |

Measured on Windows x86_64 (i9-12900K) with the official binaries; the CI
failure is the same fault on ubuntu-latest x64. The nightly result means
the fault is still on master, not something 1.13.0 picked up and master
has since lost. The magnitude depends on
the data; every row of a 144 × 16 matrix was wrong in the package's kernel.
Short recurrences pass (n = 3 and n = 4 gave 0.0 at every combination),
which points at a loop transformation applied above some trip count rather
than at the scalar lowering. `-O0` is the plain-Julia result; a
precompiled package image keeps its `-O2` code, so passing `-O0` to a script
that loads the package does not test this.

A 1-D form of the same recurrence, `x[i] -= a[i] * x[i-1]`, came out
correct on 1.13.0; the two-dimensional indexing is part of the trigger.

Also observed on **1.12.7**, in one run of the same script: the checked
variant (no `@inbounds`) with `a` wrapped in `Base.Experimental.Const` gave
a wrong result (0.047) while the `@inbounds` variant was correct. That is the
shape of JuliaGPU/KernelAbstractions.jl#652 (open; `@Const` plus compound
assignment in a loop on the CPU backend, Julia 1.11 and 1.12). So the scope
has been unreliable since 1.11 in one form and 1.13.0 added a second form
that needs no `Const` at all.

## How it reached the package

KernelAbstractions 0.9.42 wraps the CPU-backend body of a kernel in
`Expr(:aliasscope)` ... `Expr(:popaliasscope)` when, and only when, the
kernel declares a `@Const` argument (`src/macros.jl`; the scope was dropped
for the other kernels in KernelAbstractions PR #653 after a miscompile of the
same family). The ten kernels of `src/lines_device.jl` declared their
coefficient arrays and input fields `@Const`, and their bodies are
`@inbounds`. The Thomas sweep (`_dev_thomas_kernel!`) is the recurrence above
with `B[l, i]` and `lmul[i]`; every element of every line came out wrong.
The banded sweep carries the same dependency through memory and happened to
compile correctly, as did the fill and scatter kernels, which accumulate in
registers.

The serial `device line solves` testset passed on the same Julia because
julia-runtest runs the suite with `--check-bounds=yes`; the MPI leg runs
without it and failed. Neither the pointwise kernels (`pointwise_ka!`
bodies take plain arguments, no `@Const`) nor the host `@threaded` loops are
exposed, so the CPU production path is unaffected. A GPU backend compiles
through its own pipeline, where `@Const` marks loads read-only; the fault is
specific to the CPU emulation the test suite relies on for bitwise checks.

## What was done

`@Const` was removed from every kernel in `src/lines_device.jl`, so
KernelAbstractions emits no alias scope for them on the CPU backend. With
that change the `device line solves` and `staged device exchange` phases pass
at np = 2 under 1.13.0 (17/17), and the isolated Thomas kernel reproduces
`solve_cols!` bitwise again. The alternative, keeping `@Const` and carrying
each recurrence in a register so that no iteration loads what the previous
one stored, would sidestep this instance but not the family; the banded
elimination is a rank-q update that cannot be carried in one register, and
KernelAbstractions.jl#652 shows the scope failing on other shapes.

The cost on a GPU backend is the loss of the read-only load hint on the
coefficient vectors and the input field; it has not been measured
(`probes/device_floors.jl`, line-solve section, is the instrument).

## Reproduction recipe for the package

1. `juliaup add 1.13`; instantiate a checkout under `julia +1.13`.
2. `mpiexec -n 2 julia +1.13 --project=. -t 1 test/mpi_tests.jl "phases=device line solves"`
   fails 3/9 at the commit before the fix (a1d7660) and passes 9/9 after.
3. The same phase under `--check-bounds=yes` passes at either commit.

## Relation to the known issue

JuliaLang/julia#60029 (open, labels `regression 1.11`, `fixed on master`)
reports a different shape of the same construct: an accumulation
`output[I] += input[i, k]` inside `@aliasscope` with bounds checks on,
wrong on 1.11 and 1.12. The maintainers bisected it to the `Memory` PR
(#51319) and found it fixed by the pipeline change #52850, which is in
1.13. Both reproducers were run here on both versions:

| reproducer | 1.12.7 | 1.13.0 | 1.14 nightly |
|---|---|---|---|
| #60029 (accumulate, bounds checked) | wrong (4.97) | correct | correct |
| this report (`@inbounds` recurrence) | correct | wrong (0.30) | wrong (0.61) |
| KernelAbstractions kernel below, `@Const` | correct | wrong (0.08) | wrong (0.45) |
| same kernel without `@Const` | correct | correct | correct |

So 1.13.0 fixed the reported shape and introduced this one, and master
still carries it. No issue or pull request in JuliaLang/julia mentions
`aliasscope` after #60029, and the KernelAbstractions thread (#652) has no
1.13 report.

## Draft issue for JuliaLang/julia

**Title:** `@aliasscope` miscompiles a loop-carried recurrence under
`@inbounds` on 1.13.0 (regression from 1.12.7)

**Body:**

A store followed by a load of the same array element on the next loop
iteration produces wrong values inside `Base.Experimental.@aliasscope` when
bounds checks are elided. No `Base.Experimental.Const` array is involved,
so the scope should assert nothing about the arrays in the loop. 1.12.7 is
correct; 1.13.0 and the current nightly (1.14.0-DEV.3172) are wrong at the
default optimization level and correct at `-O0` or with
`--check-bounds=yes`.

```julia
using Base.Experimental: @aliasscope

function fwd!(B, a, n)
    for l in axes(B, 1)
        @aliasscope begin
            @inbounds for i in 2:n
                B[l, i] -= a[i] * B[l, i-1]
            end
        end
    end
    return B
end

function fwd_plain!(B, a, n)
    for l in axes(B, 1), i in 2:n
        B[l, i] -= a[i] * B[l, i-1]
    end
    return B
end

B = rand(1, 10); a = rand(10)
maximum(abs.(fwd!(copy(B), a, 10) .- fwd_plain!(copy(B), a, 10)))
# 1.12.7: 0.0     1.13.0: ~0.3     1.14.0-DEV.3172: ~0.6
# (data dependent; every element wrong)
```

Observations:

- Wrong for `n = 10` and above; `n = 3` and `n = 4` are correct, so a
  loop transformation above some trip count is involved rather than the
  scalar lowering.
- The 1-D form `x[i] -= a[i] * x[i-1]` is correct on 1.13.0; the second
  index is part of the trigger.
- `-O0` and `--check-bounds=yes` are both correct on 1.13.0.
- Reproduced on Windows x86_64 (official binary) and on the GitHub
  `ubuntu-latest` x64 runner.

This is the same construct as #60029 but the opposite shape: that
reproducer (accumulation through memory, bounds checked) is wrong on 1.12.7
and correct on 1.13.0, while this one is correct on 1.12.7 and wrong on
1.13.0.

How it is reached in practice: KernelAbstractions.jl wraps the CPU-backend
body of every kernel that declares a `@Const` argument in `@aliasscope`
(`src/macros.jl`, `Expr(:aliasscope)` ... `Expr(:popaliasscope)`), and
kernel bodies are commonly `@inbounds`. A Thomas sweep written as a kernel
therefore gives wrong results on 1.13.0, which is how this was found (a
bitwise comparison against the host implementation). Removing `@Const`
removes the scope and the fault. See JuliaGPU/KernelAbstractions.jl#652
for the 1.11/1.12 side of the same exposure.

```julia
using KernelAbstractions   # 0.9.42

# One work-item per row; each row is a forward recurrence carried through B.
@kernel function fwd_kernel!(B, @Const(a), n)
    l = @index(Global, Linear)
    @inbounds for i in 2:n
        B[l, i] -= a[i] * B[l, i-1]
    end
end

function fwd_plain!(B, a, n)
    for l in axes(B, 1), i in 2:n
        B[l, i] -= a[i] * B[l, i-1]
    end
    return B
end

L, n = 4, 16
B = rand(L, n); a = rand(n)
Bk = copy(B)
fwd_kernel!(CPU())(Bk, a, n; ndrange=L)
KernelAbstractions.synchronize(CPU())
maximum(abs.(Bk .- fwd_plain!(copy(B), a, n)))
# 1.12.7: 0.0     1.13.0: ~0.08     1.14.0-DEV.3172: ~0.45 (data dependent)
# Drop @Const(a) -> a, or run with --check-bounds=yes: 0.0 on both.
```

```
Julia Version 1.13.0
Commit d1c37793dd (2026-09-09 19:00 UTC)
  OS: Windows (x86_64-w64-mingw32)
  CPU: 24 × 12th Gen Intel(R) Core(TM) i9-12900K
  LLVM: libLLVM-20.1.8 (ORCJIT, alderlake)

Julia Version 1.14.0-DEV.3172 (also wrong)
Commit 639e33feaf (2026-09-12 10:20 UTC)
  LLVM: libLLVM-22.1.8 (ORCJIT, alderlake)

Julia Version 1.12.7 (correct)
Commit 6d172b025e (2026-08-15 08:05 UTC)
  LLVM: libLLVM-18.1.7 (ORCJIT, alderlake)
```
