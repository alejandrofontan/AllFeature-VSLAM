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
    AF_INFO("track: heartbeat | frame=" << current_frame_.frame_id
            << " inliers=" << num_inlier_matches_
            << " localPts=" << local_points_.size()
            << " KFs=" << map_->keyframes_in_map()
            << " mapPts=" << map_->map_points_in_map());
    std::cout.flush(); // stdout is fully buffered under the runner's redirect
#endif
}

// Cumulative per-stage timing histograms, printed through the AF_PROFILE sink.
void Tracking::log_profile()
{
#ifdef PROFILING_EXHAUSTIVE
    AF_PROFILE_BEGIN("Tracking Profiling");
    AF_PROFILE_FIELD(resize_times_,      "Resize Image");
    AF_PROFILE_FIELD(frame_times_,       "Frame Creation");
    AF_PROFILE_FIELD(tracking_times_,    "Tracking");
    AF_PROFILE_FIELD(track_ref_times_,   "  Track Ref");
    AF_PROFILE_FIELD(pose_opt_times_,    "  Pose Optimization");
    AF_PROFILE_FIELD(local_map_times_,   "  Track Local Map");
    AF_PROFILE_FIELD(grab_image_times_, "Grab Image");
    AF_PROFILE_END();
#endif
}

bool Tracking::run_tracking_stage(const std::function<bool()>& stage)
{
    try {
        return stage();
    }
    catch(const TrackingLostException& e) {
        AF_WARN("Tracking lost — " << e.what());
        return false;
    }
}

} // namespace AF_VSLAM
