/**
 * Auxiliary plumbing for Tracking::Track(), kept out of Tracking.cc so the
 * per-frame flow there reads uninterrupted.
 *
 * TrackProfiler: stage timing and slow-frame reporting for Track() (hiccup
 * diagnosis; report threshold ~3 frame periods at 20 fps). All the
 * PROFILING_EXHAUSTIVE conditioning lives here, so Track() itself reads as
 * plain calls that compile to empty no-ops when profiling is off.
 */
#ifndef AF_VSLAM_TRACKING_AUX_H
#define AF_VSLAM_TRACKING_AUX_H

#include <chrono>
#include <map>

#include "Definitions.h"
#include "Types.h"
#include "afvslam_log.hpp"

namespace AF_VSLAM {

struct TrackProfiler
{
#ifdef PROFILING_EXHAUSTIVE
    static constexpr double SLOW_FRAME_WARN_MS = 150.0;
    using Clock = std::chrono::steady_clock;
    static double ms(Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }
    Clock::time_point t_start{Clock::now()};
    Clock::time_point t_lock{}, t_track_ref{}, t_wait_start{};
    double ms_lock_wait{0.0}, ms_track_ref{0.0}, ms_local_map{0.0}, ms_emergency_wait{0.0};
#endif

    void lock_acquired() {
#ifdef PROFILING_EXHAUSTIVE
        t_lock = Clock::now();
        ms_lock_wait = ms(t_start, t_lock);
#endif
    }

    void track_ref_done() {
#ifdef PROFILING_EXHAUSTIVE
        t_track_ref = Clock::now();
#endif
    }

    void local_map_done([[maybe_unused]] std::map<int, int>& local_map_times) {
#ifdef PROFILING_EXHAUSTIVE
        ms_track_ref = ms(t_lock, t_track_ref);
        ms_local_map = ms(t_track_ref, Clock::now());
        local_map_times[int(ms_local_map)]++;
#endif
    }

    void emergency_wait_begin() {
#ifdef PROFILING_EXHAUSTIVE
        t_wait_start = Clock::now();
#endif
    }

    void emergency_wait_end() {
#ifdef PROFILING_EXHAUSTIVE
        ms_emergency_wait = ms(t_wait_start, Clock::now());
#endif
    }

    // Whenever this frame stalled visibly, say where the time went. "other"
    // covers keyframe decision/creation, drawer updates, and pose bookkeeping.
    void report([[maybe_unused]] FrameId frame_id) {
#ifdef PROFILING_EXHAUSTIVE
        const double ms_total = ms(t_start, Clock::now());
        if (ms_total > SLOW_FRAME_WARN_MS)
        {
            AF_WARN("Track: slow frame, " << int(ms_total) << " ms"
                    << " (mapMutexWait=" << int(ms_lock_wait)
                    << ", trackRef=" << int(ms_track_ref)
                    << ", localMap=" << int(ms_local_map)
                    << ", emergencyWait=" << int(ms_emergency_wait)
                    << ", other=" << int(ms_total - ms_lock_wait - ms_track_ref - ms_local_map - ms_emergency_wait)
                    << ") | frame=" << frame_id);
        }
#endif
    }
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_TRACKING_AUX_H
