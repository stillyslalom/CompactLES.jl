// Julia-free rung of the wait-stall reproducer ladder
// (reference/rocm_wait_stall_report.md): a plain HIP loop — one small
// kernel launch plus hipStreamSynchronize per call — with an optional
// number of extra dormant host threads, since the Julia-side measurements
// show the stall requires a process with more than one default-pool
// thread. If this stalls with dormant threads, the problem is any
// multithreaded process and belongs to ROCR/the driver; if it never
// stalls, the trigger involves the Julia runtime specifically (its GC and
// scheduler threads, signal use, or foreign-call transitions).
//
//   hipcc -O2 --offload-arch=native -o stall_mwe stall_mwe.cpp
//   ./stall_mwe [seconds=120] [iters=20000] [threads=7] [busy=0]
//
// The arch flag is not optional: hipcc's default offload target is an
// older gfx, a mismatched code object makes every launch a silent no-op,
// and hipStreamSynchronize still returns success — the loop then measures
// nothing at ~0.2 us/call (observed). The hipGetLastError check below
// turns that into a hard failure; the calibration line is the sanity
// check that a kernel actually ran (expect tens of microseconds, not
// sub-microsecond). If --offload-arch=native is unavailable, name the
// target explicitly (gfx942 on MI300A).
//
// threads: extra host threads to spawn; they sleep (busy=0) or spin
// (busy=1). 7 dormant threads mirrors julia -t 8. Compare seconds-long
// runs at threads=0 and threads=7; a stalled window reads a median near
// 13 ms (ROCR's polling sleep quantum) against a baseline set by iters.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define HIP_CHECK(expr)                                                  \
    do {                                                                 \
        hipError_t err_ = (expr);                                        \
        if (err_ != hipSuccess) {                                        \
            std::fprintf(stderr, "HIP error %s at line %d: %s\n",        \
                         hipGetErrorName(err_), __LINE__,                \
                         hipGetErrorString(err_));                       \
            std::exit(1);                                                \
        }                                                                \
    } while (0)

__global__ void spin_kernel(float* x, int iters) {
    int i = threadIdx.x;
    float acc = x[i];
    for (int k = 0; k < iters; ++k) acc = fmaf(acc, 0.9999f, 1e-7f);
    x[i] = acc;
}

int main(int argc, char** argv) {
    double seconds = argc > 1 ? std::atof(argv[1]) : 120.0;
    int iters = argc > 2 ? std::atoi(argv[2]) : 20000;
    int nthreads = argc > 3 ? std::atoi(argv[3]) : 7;
    int busy = argc > 4 ? std::atoi(argv[4]) : 0;

    std::atomic<bool> stop{false};
    std::vector<std::thread> extras;
    for (int t = 0; t < nthreads; ++t) {
        extras.emplace_back([&stop, busy]() {
            volatile double sink = 0.5;
            while (!stop.load(std::memory_order_relaxed)) {
                if (busy)
                    for (int k = 0; k < 100000; ++k) sink = sink * 0.999999 + 1e-9;
                else
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }

    const int n = 256;
    float* x = nullptr;
    HIP_CHECK(hipMalloc(&x, n * sizeof(float)));
    HIP_CHECK(hipMemset(x, 0, n * sizeof(float)));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    auto call = [&]() {
        hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(n), 0, 0, x, iters);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(0));
    };
    call();
    call();
    double cal = 1e9;
    for (int k = 0; k < 50; ++k) {
        auto a = std::chrono::steady_clock::now();
        call();
        auto b = std::chrono::steady_clock::now();
        cal = std::min(cal, std::chrono::duration<double>(b - a).count());
    }
    const char* hsaint = std::getenv("HSA_ENABLE_INTERRUPT");
    std::printf("device %s  iters %d  extra threads %d (%s)  "
                "HSA_ENABLE_INTERRUPT=%s  calibration %.3f ms/call\n",
                prop.name, iters, nthreads, busy ? "busy" : "dormant",
                hsaint ? hsaint : "(unset)", 1e3 * cal);

    std::vector<double> times;
    times.reserve(static_cast<size_t>(2e4 * seconds));
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        auto a = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(a - t0).count() > seconds) break;
        call();
        auto b = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(b - a).count());
    }
    stop.store(true);
    for (auto& th : extras) th.join();

    std::vector<double> s(times);
    std::sort(s.begin(), s.end());
    size_t nc = s.size();
    double med = s[nc / 2];
    double p99 = s[static_cast<size_t>(0.99 * nc)];
    std::printf("%zu calls: median %.3f ms  p99 %.3f ms  max %.3f ms\n",
                nc, 1e3 * med, 1e3 * p99, 1e3 * s[nc - 1]);

    // Slow threshold 1 ms absolute (see the Julia rung for why relative
    // thresholds invert); episodes are maximal runs of consecutive slow
    // calls.
    double thr = 1e-3, stall_t = 0.0, cur = 0.0, longest = 0.0;
    long n_slow = 0;
    for (double t : times) {
        if (t > thr) {
            ++n_slow;
            cur += t;
            stall_t += t;
            longest = std::max(longest, cur);
        } else {
            cur = 0.0;
        }
    }
    double total = 0.0;
    for (double t : times) total += t;
    std::printf("slow calls >1 ms: %ld, %.2f s total (%.1f%% of watch); "
                "longest episode %.2f s\n",
                n_slow, stall_t, 100.0 * stall_t / total, longest);

    const double edges[] = {0.25e-3, 0.5e-3, 1e-3, 2e-3, 5e-3, 10e-3};
    const char* labels[] = {"<=0.25ms", "0.25-0.5", "0.5-1",
                            "1-2", "2-5", "5-10", ">10ms"};
    long counts[7] = {0};
    for (double t : times) {
        int b = 0;
        while (b < 6 && t > edges[b]) ++b;
        ++counts[b];
    }
    for (int b = 0; b < 7; ++b)
        if (counts[b]) std::printf("  %s: %ld", labels[b], counts[b]);
    std::printf("\n");
    HIP_CHECK(hipFree(x));
    return 0;
}
