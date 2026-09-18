/**
 * Auxiliary LoopClosing members kept out of LoopClosing.cc so the loop pipeline
 * there (detection, Sim3 verification, correction, global BA) reads
 * uninterrupted: parameter loading and thread synchronization (keyframe queue,
 * global-BA status, reset and finish protocols; see the protocol comments in
 * LoopClosing.h).
 */
#include "LoopClosing.h"

#include <chrono>
#include <mutex>
#include <thread>

namespace AF_VSLAM
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Parameters
LoopClosingParameters LoopClosing::params{};

void LoopClosing::LoadParameters(const cv::FileStorage& fSettings)
{
    auto read_if_present = [&fSettings](const char* key, auto& field)
    {
        const cv::FileNode node = fSettings[key];
        if(!node.empty())
            node >> field;
    };

    read_if_present("LoopClosing.CovisibilityConsistencyThreshold", params.covisibility_consistency_threshold);
    read_if_present("LoopClosing.MinKeyframesBetweenLoops", params.min_keyframes_between_loops);
    read_if_present("LoopClosing.Sim3MinMatches", params.sim3_min_matches);
    read_if_present("LoopClosing.Sim3MinInliers", params.sim3_min_inliers);
    read_if_present("LoopClosing.LoopMinMatches", params.loop_min_matches);
    read_if_present("LoopClosing.FuseRadius", params.fuse_radius);
    read_if_present("LoopClosing.GbaIterations", params.gba_iterations);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Keyframe queue
void LoopClosing::insert_keyframe(const Keyframe& keyframe)
{
    // Without an active VPR backend this thread never runs (System's constructor): a
    // queued keyframe would never be drained and would stay alive, culled or not.
    if(!place_recognition_->is_active())
        return;

    std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
    new_keyframes_.push_back(keyframe);
}

bool LoopClosing::has_new_keyframes() const
{
    std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
    return !new_keyframes_.empty();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Global bundle adjustment status
bool LoopClosing::is_gba_running() const
{
    std::lock_guard<std::mutex> lock(gba_mutex_);
    return gba_running_;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Reset protocol
void LoopClosing::request_reset()
{
    // Without an active VPR backend this thread never runs, so nobody would ever clear
    // the request (the caller would spin forever); there is nothing to reset either.
    if(!place_recognition_->is_active())
        return;

    {
        std::lock_guard<std::mutex> lock(reset_mutex_);
        reset_requested_ = true;
    }
    // Block until the Loop Closing thread has performed the reset (reset_if_requested)
    while(is_reset_requested())
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

bool LoopClosing::is_reset_requested() const
{
    std::lock_guard<std::mutex> lock(reset_mutex_);
    return reset_requested_;
}

void LoopClosing::reset_if_requested()
{
    std::lock_guard<std::mutex> lock(reset_mutex_);
    if(!reset_requested_)
        return;

    {
        std::lock_guard<std::mutex> queue_lock(new_keyframes_mutex_);
        new_keyframes_.clear();
    }
    last_loop_keyframe_id_ = 0;

    // Loop-detection state of the wiped map. Keyframe ids restart at 0 after a reset, so
    // stale consistency groups (keyed by id) could vote for the new map's first
    // candidates; the vectors would also keep the old map's keyframes and points alive.
    consistent_groups_.clear();
    loop_candidates_.clear();
    loop_matched_points_.clear();
    loop_map_points_.clear();
    current_keyframe_.reset();
    matched_keyframe_.reset();

    reset_requested_ = false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Finish protocol
void LoopClosing::request_finish()
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    finish_requested_ = true;
}

bool LoopClosing::is_finish_requested() const
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    return finish_requested_;
}

void LoopClosing::set_finished()
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    finished_ = true;
}

bool LoopClosing::is_finished() const
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    return finished_;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace AF_VSLAM
