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

| threads | jobs | rank-windows | sustained episodes | stalled fraction per rank |
|---|---|---|---|---|
| `-t 8` | 1 | 4 (240 rank-s) | ~10 (up to 41.8 s) | 88.9% / 27.3% / 1.5% / 34.8% |
| `-t 1` | 2 | 8 (480 rank-s) | 0 | 1.7–1.8% on every rank |

At the `-t 8` episode rate, eight consecutive clean 60 s windows has
probability on the order of e⁻²⁰. The stall mode requires a multithreaded
Julia process, yet survives the removal of every Julia-level wait
construct — consistent with an interaction between the multithreaded
runtime (thread parking/futex behavior, foreign-call transitions) and
ROCR's interrupt-driven signal waits, whose wakeup cadence is the natural
source of the millisecond ticks. Untested hypothesis; discriminating
experiments below.

An unrelated, benign signature for contrast: at `-t 1` every rank shows
the same deterministic pair of startup events (~0.15 s near 1 s, ~0.11 s
near 7 s, plus one ~150 ms call) reproducible across jobs; the same ~7 s
event appears at `-t 8`. This is not the stall mode.

## Open experiments

1. Thread threshold: the same watch pair at `-t 2` and `-t 4`.
2. ROCR wait mode: busy-wait signal polling instead of interrupt waits
   (`HSA_ENABLE_INTERRUPT=0` is the historical spelling; verify against
   ROCm 6.4 documentation) — if the 13/26 ms ticks are interrupt wakeups,
   this should remove or transform them.
3. `rocprofv2` sys-trace over a watch that contains a stall: does the time
   sit in kernel execution, in gaps between kernels, or inside the HSA
   signal wait?
4. LC ticket: driver/firmware state and node health are visible to LC and
   not to us, and MI300A wait-latency issues may be known.

## Practical guidance until resolved

Run device-resident jobs at `-t 1` per rank. The device step does not need
host threads (measured TGV s/step is unchanged), and no wall-clock number
from a multithreaded process is trustworthy without a stall watch beside
it.
