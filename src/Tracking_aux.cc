/**
 * Auxiliary Tracking members kept out of Tracking.cc so the per-frame flow
 * there reads uninterrupted (companion to include/Tracking_aux.h).
 */
#include "Tracking.h"

#include <iostream>

#include "afvslam_log.hpp"

namespace AF_VSLAM
{

// Low-rate heartbeat so post-mortems can see the inlier/map trend leading
// into a tracking loss, not just the loss line itself.
void Tracking::log_heartbeat()
{
#ifdef PROFILING_EXHAUSTIVE
    constexpr int HEARTBEAT_PERIOD_FRAMES = 100;
    if(current_frame_.frame_id % HEARTBEAT_PERIOD_FRAMES != 0)
        return;
    AF_INFO("Track heartbeat | frame=" << current_frame_.frame_id
            << " inliers=" << num_inlier_matches_
            << " localPts=" << local_points_.size()
            << " KFs=" << map_->keyframes_in_map()
            << " mapPts=" << map_->map_points_in_map());
    std::cout.flush(); // stdout is fully buffered under the runner's redirect
#endif
}

bool Tracking::run_tracking_stage(const std::function<bool()>& stage)
{
    try {
        return stage();
    }
    catch(const TrackingLostException& e) {
        last_tracking_lost_reason_ = e.what();
        AF_WARN("Tracking lost — " << last_tracking_lost_reason_);
        return false;
    }
}

} // namespace AF_VSLAM
