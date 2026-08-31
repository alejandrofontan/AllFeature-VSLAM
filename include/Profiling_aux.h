/**
 * Shared profiling primitives for the per-thread profilers (Tracking_aux.h,
 * LocalMapping_aux.h). Everything here compiles to a no-op when
 * PROFILING_EXHAUSTIVE is off.
 */
#ifndef AF_VSLAM_PROFILING_AUX_H
#define AF_VSLAM_PROFILING_AUX_H

#include <chrono>
#include <map>

#include "Definitions.h"

namespace AF_VSLAM {

// One-shot stage timer for the PROFILING_EXHAUSTIVE histograms: record()
// buckets the milliseconds since construction (or since the previous record)
// into the given histogram. Compiles to a no-op when profiling is off.
struct StageTimer
{
#ifdef PROFILING_EXHAUSTIVE
    using Clock = std::chrono::steady_clock;
    Clock::time_point t_last{Clock::now()};
#endif

    void record([[maybe_unused]] std::map<int, int>& histogram) {
#ifdef PROFILING_EXHAUSTIVE
        const auto now = Clock::now();
        histogram[int(std::chrono::duration<double, std::milli>(now - t_last).count())]++;
        t_last = now;
#endif
    }
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_PROFILING_AUX_H
