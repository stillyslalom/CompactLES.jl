# Sustained millisecond-quantized HIP wait stalls on MI300A under multithreaded Julia

**Status:** characterization for an LC ticket and/or upstream report
(AMDGPU.jl, ROCR). Raw session transcripts:
`bench/logs/rzadams_20260819.txt`, `bench/logs/rzadams_20260819_floors.txt`.
Reproducer: `bench/device_floors.jl only=watch watch=60` (times a fixed
small compact-solve apply — several kernel launches, a device→host and a
host→device copy, one stream synchronize — continuously and reports
per-call statistics and episode structure).

## Symptom

A process spontaneously enters a state in which every device wait costs an
integer number of milliseconds instead of the ~0.14 ms baseline, holds that
state for seconds to at least 42 s, then recovers. While stalled, the
per-call distribution collapses onto 13 ms and 26 ms (one fully stalled
30 s window measured a median of exactly 13.000 ms and a p99 of 26.007 ms),
a ~100× per-call throughput drop.

- Entry and exit are spontaneous; entry offsets scatter across a 60 s
  window (4–41 s observed). Two multi-second episodes were observed
  back-to-back with at most one fast call between them, so exits can be
  single-call blips inside a longer stalled state. One episode ran to the
  measurement boundary (duration censored), so the dwell distribution is
  heavy-tailed with unknown maximum.
- State is per-process: ranks on sibling APUs of the same node stall
  independently (one rank 89% stalled while a sibling ran 1.5%).
- Indifferent to workload details: strikes Float64 and Float32,
  tridiagonal and pentadiagonal solves, every sweep dimension, isolated
  applies and whole solver runs.

## Environment

- LLNL rzadams: 4× AMD MI300A per node, `gfx942:sramecc+:xnack-`,
  ROCm 6.4.3, TOSS/flux, launched `flux run -N1 -n4 --exclusive`
  (mpibind: one APU + one socket per rank).
- Julia 1.12.7, AMDGPU.jl v2.7.2, Cray MPICH (host-staged MPI; the
  reproducer's watch section performs no inter-rank communication).
- Not reproduced on the one other machine measured: Windows 11 /
  RX 6800 XT (gfx1030) / HIP SDK / Julia 1.11.4 / same AMDGPU.jl — but OS,
  driver, GPU, and Julia version all differ, so this discriminates
  nothing by itself.

## Ruled out by measurement

1. **Garbage collection.** A fully stalled 30 s window recorded 0 ms of GC
   time; clean windows on sibling ranks recorded 400–760 ms. No
   correlation in either direction.
2. **AMDGPU.jl's task-based synchronize fallback.** With the package
   preference `nonblocking_synchronization = false` (verified active by
   the recompile on entry and by the launch+sync floor dropping from
   ~25 µs to 18–19 µs), every wait routes through plain
   `hipStreamSynchronize` with no Julia tasks or libuv involvement — and
   the stalls persisted with identical quantization (one rank 96.9% of its
   watch at p99 26.004 ms). The quantized wait is below the Julia layer.
3. **Workload dependence.** See above; also the artifact scheme/precision
   patterns of small samples (a "C6 Float64 dims 1–2 defect", a "26×
   Float64/Float32 step ratio") dissolved under larger samples into this
   one mode striking arbitrary cells.

## The one strong correlate: Julia thread count

Watch-only incidence runs, 60 s per rank, 4 ranks per job, identical
nodes/launch line, `nonblocking_synchronization = false` throughout:

| threads | jobs | rank-windows | windows with a sustained episode | worst rank |
|---|---|---|---|---|
| `-t 1` | 2 | 8 (480 rank-s) | 0 | 1.8% slow |
| `-t 2` | 1 | 4 | 1 | 78.9% stalled (38.6 s + 8.5 s back-to-back, censored) |
| `-t 4` | 1 | 4 | 1 | 83.6% stalled (49.9 s episode, censored) |
| `-t 8` | 1 | 4 | 3 | 88.9% stalled (41.8 s episode) |

At the `-t 8` episode rate, eight consecutive clean 60 s windows has
probability on the order of e⁻²⁰; the mode exists at every thread count
above 1 and incidence grows with thread count. Note that a Julia 1.12
`-t 1` process already carries an interactive thread (`versioninfo`
reports "1 default, 1 interactive"), so the discriminator is more than
one default-pool thread, not more than one OS thread. The stall requires
a multithreaded Julia process yet survives the removal of every
Julia-level wait construct, placing it in ROCR's signal waits — confirmed
by the wait-mode experiment in the next section, which reproduces the
stalled cadence deterministically by forcing the polling wait class. Once entered, the state
typically holds to the end of the measurement window (multiple episodes
censored at the boundary), so dwell has no observed upper bound. Stalled
ranks report *less* GC time than clean siblings (fewer calls, fewer
allocations): GC is a consequence of the call rate here, not a cause.

An unrelated, benign signature for contrast: at `-t 1` every rank shows
the same deterministic pair of startup events (~0.15 s near 1 s, ~0.11 s
near 7 s, plus one ~150 ms call) reproducible across jobs; the same ~7 s
event appears at `-t 8`. This is not the stall mode.

## The stall state identified: ROCR's polling wait

`HSA_ENABLE_INTERRUPT=0` switches completion-signal detection from
interrupts (`InterruptSignal`) to memory-based polling (`BusyWaitSignal`).
Running the watch under it (two 4-rank jobs, `-t 8`) reproduced the stall
state **deterministically**: six of eight rank-windows spent 100.0% of the
watch stalled from t = 0 with medians of exactly 12.999–13.001 ms and p99
at 26.00 ms — the identical cadence the intermittent episodes show under
default interrupt mode — and the other two windows read 94.9% and 99.2%.
The mixed windows interleave thousands of sub-0.25 ms calls, so even
forced polling sometimes completes within its initial spin; ~13 ms is the
first sleep quantum of the polling wait, 26 ms two quanta.

The finding therefore reduces to: **under default interrupt mode, a
process's signal waits intermittently degrade to the polling-backoff
cadence for seconds to minutes** — interrupt wakeups stop arriving
promptly (or stop being waited on) and waiters ride the polling fallback —
and the degradation requires more than one Julia default-pool thread.
`HSA_ENABLE_INTERRUPT=0` is the pathology made permanent, not a
workaround.

## Remaining steps: the reproducer ladder

Isolate to the smallest reproducer before filing (a research code will
not get driver-team priority on a whole-application report).

1. **`bench/stall_mwe.jl`** — AMDGPU.jl alone, no CompactLES, no
   KernelAbstractions, no MPI: one trivial kernel plus one stream
   synchronize per call (`mode=kernel`), with rungs adding a second
   launch, a device-to-host copy, and a round trip (`kernel2`, `copy`,
   `full`), and `sync=direct` bypassing all AMDGPU.jl wait logic through
   a raw `hipStreamSynchronize` ccall. **Measured: the bottom rung does
   not reproduce** — two 120 s windows at `-t 8` and one at `-t 1`,
   single rank, all clean (≤1 slow call per 256k). The differences from
   the reliably stalling configuration are process count (every reliable
   reproduction was a 4-rank job) and call content (the solver call is
   many short kernels plus copies in an MPI-initialized process, with
   waits of tens of microseconds; the MWE is one 0.47 ms kernel whose
   sync always waits on a single signal — and several stalled windows
   were stalled from t = 0, implicating startup activity such as MPI
   init or the kernel-compilation burst). **The 2×2 is measured and
   decided**: the floors watch stalls at `-n1` (29.7 s episode, 25.7% of
   a 120 s window) and the MWE stays clean at `-n4` (four ranks × 120 s,
   ≤2 slow calls each). Process count is neither necessary nor
   sufficient; the ingredient is call content or process history. One
   confound dominates the corners: every run observed to stall had MPI
   initialized (Cray MPICH — progress threads, memory-registration
   hooks, GPU-aware transport plumbing), and every clean MWE run did
   not. **`mpi=1` measured clean** (120 s at `-t 8`, 1 slow call in
   255k): MPI initialization alone is insufficient. The leading rung is
   now `alloc=N`, which allocates N heap bytes per call to drive the
   garbage collector the way the real call does (its device-to-host
   staging materializes a host array every apply, ~1 s of GC per 120 s
   window; the clean MWE rungs allocate nothing per call). The
   hypothesis it tests: GC does not sustain the stalled state (a fully
   stalled window measured zero GC time), but a collection's
   inter-thread stop signals could be the *entry* event that knocks
   ROCR's event wait into its degraded mode — which would also explain
   the more-than-one-default-thread gate, since stopping the world only
   signals other threads when other threads exist. Behind it in the
   queue: `mode=full` (the copy path), `work=2000` (tens-of-microsecond
   waits racing signal completion), and the startup compile burst.
2. **`bench/stall_mwe.cpp`** — the Julia-free rung: the same loop in
   plain HIP with N extra dormant (or busy) host threads
   (`hipcc -O2 -o stall_mwe stall_mwe.cpp`; `./stall_mwe 120 20000 7`).
   This is the decisive fork: if it stalls with extra threads, the
   trigger is any multithreaded process and the ticket is unambiguously
   ROCR/driver territory; if it never stalls, the trigger involves the
   Julia runtime specifically (GC and scheduler threads, signal use,
   foreign-call transitions) and the report goes to AMDGPU.jl/Julia
   upstream first, with LC secondary.
3. File with whichever party the fork selects, attaching the winning MWE,
   this report, and the two logs. Optional corroboration if asked: a
   `rocprofv2` sys-trace over a stalled window should show the gap inside
   the HSA signal wait rather than in kernel execution, and the ~13 ms
   constant should be findable in ROCR's `BusyWaitSignal` implementation.

## Practical guidance until resolved

Run device-resident jobs at `-t 1` per rank. The device step does not need
host threads (measured TGV s/step is unchanged), and no wall-clock number
from a multithreaded process is trustworthy without a stall watch beside
it.
