/**
 * Auxiliary plumbing for LocalMapping::run(), kept out of LocalMapping.cc so
 * the per-keyframe flow there reads uninterrupted (companion to
 * src/LocalMapping_aux.cc, which holds the auxiliary member definitions).
 *
 * LocalMappingProfiler: total per-keyframe mapping time. Always measured — its
 * histogram feeds the viewer's median-time display, so it must work in
 * non-profiling builds too (same rationale as Tracking's GrabProfiler total).
 * The per-stage histograms are recorded at the call sites through StageTimer,
 * which is PROFILING_EXHAUSTIVE-only.
 */
#ifndef AF_VSLAM_LOCALMAPPING_AUX_H
#define AF_VSLAM_LOCALMAPPING_AUX_H

#include <chrono>
#include <map>

#include "Profiling_aux.h"

namespace AF_VSLAM {

struct LocalMappingProfiler
{
    using Clock = std::chrono::steady_clock;
    Clock::time_point t_start{Clock::now()};

    void iteration_done(std::map<int, int>& histogram) const {
        histogram[int(std::chrono::duration<double, std::milli>(Clock::now() - t_start).count())]++;
    }
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_LOCALMAPPING_AUX_H
