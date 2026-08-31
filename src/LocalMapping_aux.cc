/**
 * Auxiliary LocalMapping members kept out of LocalMapping.cc so the
 * per-keyframe flow there reads uninterrupted (companion to
 * include/LocalMapping_aux.h): parameter loading, thread synchronization, the
 * online keyframe VPR similarity matrix, and profiling output.
 */
#include "LocalMapping.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>

#include "Utils.h"
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

LocalMappingParameters LocalMapping::params{};

void LocalMapping::LoadParameters(const cv::FileStorage &fSettings)
{
    auto read_if_present = [&fSettings](const char* key, auto& field)
    {
        const cv::FileNode node = fSettings[key];
        if(!node.empty())
            node >> field;
    };

    read_if_present("LocalMapping.KeyframeCullingMethod", params.keyframe_culling_method);
    if(params.keyframe_culling_method != "heuristic" && params.keyframe_culling_method != "information")
        AF_ERROR("[LocalMapping] Unknown LocalMapping.KeyframeCullingMethod '" + params.keyframe_culling_method + "' (options: heuristic, information)");

    read_if_present("LocalMapping.MapPointCullingMinFoundRatio", params.map_point_culling_min_found_ratio);
    read_if_present("LocalMapping.MapPointCullingMinObservations", params.map_point_culling_min_observations);
    read_if_present("LocalMapping.MapPointCullingObservationTestAge", params.map_point_culling_observation_test_age);
    read_if_present("LocalMapping.MapPointCullingProbationAge", params.map_point_culling_probation_age);

    read_if_present("LocalMapping.SearchInNeighborsKeyframes", params.search_in_neighbors_keyframes);
    read_if_present("LocalMapping.SearchInNeighborsSecondKeyframes", params.search_in_neighbors_second_keyframes);
    read_if_present("LocalMapping.SearchInNeighborsRadius", params.search_in_neighbors_radius);

    read_if_present("LocalMapping.KeyframeCullingRedundancyRatio", params.keyframe_culling_redundancy_ratio);
    read_if_present("LocalMapping.KeyframeCullingMinObservations", params.keyframe_culling_min_observations);

    float max_unexplained = params.keyframe_culling_max_unexplained.load();
    read_if_present("LocalMapping.KeyframeCullingMaxUnexplained", max_unexplained);
    params.keyframe_culling_max_unexplained.store(max_unexplained);
    read_if_present("LocalMapping.KeyframeCullingMinAge", params.keyframe_culling_min_age);
    read_if_present("LocalMapping.KeyframeCullingMinKeyframes", params.keyframe_culling_min_keyframes);
    read_if_present("LocalMapping.KeyframeCullingScope", params.keyframe_culling_scope);
    if(params.keyframe_culling_scope != "map" && params.keyframe_culling_scope != "local")
        AF_ERROR("[LocalMapping] Unknown LocalMapping.KeyframeCullingScope '" + params.keyframe_culling_scope + "' (options: map, local)");
    read_if_present("LocalMapping.KeyframeCullingMaxPerCall", params.keyframe_culling_max_per_call);
    int centred = params.keyframe_culling_centred ? 1 : 0;   // cv::FileStorage has no bool reader
    read_if_present("LocalMapping.KeyframeCullingCentred", centred);
    params.keyframe_culling_centred = (centred != 0);
}

// ------------------------------------------------------------------------------------------
// Thread synchronization (see the protocol comments in LocalMapping.h)
// ------------------------------------------------------------------------------------------

void LocalMapping::insert_keyframe(const Keyframe& keyframe)
{
    std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
    new_keyframes_.push_back(keyframe);
}

bool LocalMapping::has_new_keyframes() const
{
    std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
    return !new_keyframes_.empty();
}

void LocalMapping::request_stop()
{
    std::lock_guard<std::mutex> lock(stop_mutex_);
    stop_requested_ = true;
}

bool LocalMapping::stop_if_requested()
{
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if(!stop_requested_ || insertion_locked_)
        return false;

    stopped_ = true;
    AF_INFO("[LocalMapping] stopped");
    std::cout.flush(); // AF_INFO's stdout is fully buffered under the runner's redirect
    return true;
}

bool LocalMapping::is_stopped() const
{
    std::lock_guard<std::mutex> lock(stop_mutex_);
    return stopped_;
}

bool LocalMapping::is_stop_requested() const
{
    std::lock_guard<std::mutex> lock(stop_mutex_);
    return stop_requested_;
}

void LocalMapping::release()
{
    // scoped_lock (deadlock-free acquisition): set_finished takes the same two mutexes in
    // the opposite order from the Local Mapping thread.
    std::scoped_lock lock(stop_mutex_, finish_mutex_);
    if(finished_)
        return;

    stopped_ = false;
    stop_requested_ = false;
    {
        std::lock_guard<std::mutex> queue_lock(new_keyframes_mutex_);
        new_keyframes_.clear();
    }
    AF_INFO("[LocalMapping] released");
    std::cout.flush();
}

bool LocalMapping::accepts_keyframes() const
{
    std::lock_guard<std::mutex> lock(accept_mutex_);
    return accept_keyframes_;
}

void LocalMapping::set_accept_keyframes(const bool accept)
{
    std::lock_guard<std::mutex> lock(accept_mutex_);
    accept_keyframes_ = accept;
}

bool LocalMapping::set_insertion_lock(const bool locked)
{
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if(locked && stopped_)
        return false;

    insertion_locked_ = locked;
    return true;
}

void LocalMapping::request_reset()
{
    {
        std::lock_guard<std::mutex> lock(reset_mutex_);
        reset_requested_ = true;
    }
    // Block until the Local Mapping thread has performed the reset (reset_if_requested)
    while(is_reset_requested())
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
}

bool LocalMapping::is_reset_requested() const
{
    std::lock_guard<std::mutex> lock(reset_mutex_);
    return reset_requested_;
}

void LocalMapping::reset_if_requested()
{
    std::lock_guard<std::mutex> lock(reset_mutex_);
    if(!reset_requested_)
        return;

    {
        std::lock_guard<std::mutex> queue_lock(new_keyframes_mutex_);
        new_keyframes_.clear();
    }
    recent_map_points_.clear();
    {
        std::lock_guard<std::mutex> vpr_lock(vpr_mutex_);
        vpr_keyframes_.clear();
        keyframe_vpr_matrix_.resize(0, 0);
    }

    local_mapping_times_.clear();
    create_new_map_points_times_.clear();
    search_in_neighbors_times_.clear();
    local_ba_times_.clear();

    reset_requested_ = false;
}

void LocalMapping::request_finish()
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    finish_requested_ = true;
}

bool LocalMapping::is_finish_requested() const
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    return finish_requested_;
}

void LocalMapping::set_finished()
{
    // Same two mutexes as release(), acquired deadlock-free
    std::scoped_lock lock(finish_mutex_, stop_mutex_);
    finished_ = true;
    stopped_ = true; // releases anyone waiting on is_stopped()
}

bool LocalMapping::is_finished() const
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    return finished_;
}

// ------------------------------------------------------------------------------------------
// Online keyframe VPR similarity matrix
// ------------------------------------------------------------------------------------------

Eigen::MatrixXf LocalMapping::keyframe_vpr_matrix() const
{
    std::lock_guard<std::mutex> lock(vpr_mutex_);
    return keyframe_vpr_matrix_;
}

std::vector<Keyframe> LocalMapping::keyframe_vpr_order() const
{
    std::lock_guard<std::mutex> lock(vpr_mutex_);
    return vpr_keyframes_;
}

bool LocalMapping::has_keyframe_vpr_matrix() const
{
    std::lock_guard<std::mutex> lock(vpr_mutex_);
    return !vpr_keyframes_.empty();
}

void LocalMapping::grow_keyframe_vpr_matrix(const Keyframe& keyframe)
{
    // The descriptor is computed by compute_global_descriptor() at the top of
    // process_new_keyframe; it is empty when the VPR backend is not image-based (bow/none).
    const std::vector<float>& descriptor = keyframe->global_descriptor;
    if(descriptor.empty())
        return;
    std::lock_guard<std::mutex> lock(vpr_mutex_);
    const int k = int(vpr_keyframes_.size());
    keyframe_vpr_matrix_.conservativeResize(k + 1, k + 1);
    for(int i = 0; i < k; i++){
        const std::vector<float>& other = vpr_keyframes_[i]->global_descriptor;
        float s = std::numeric_limits<float>::quiet_NaN();
        if(other.size() == descriptor.size()){
            double dot = 0.0;
            for(size_t d = 0; d < descriptor.size(); d++)
                dot += double(descriptor[d]) * other[d];
            s = float(dot);   // unit descriptors: dot product == cosine
        }
        keyframe_vpr_matrix_(i, k) = s;
        keyframe_vpr_matrix_(k, i) = s;
    }
    keyframe_vpr_matrix_(k, k) = 1.0f;
    vpr_keyframes_.push_back(keyframe);
}

void LocalMapping::print_keyframe_vpr_matrix() const
{
    std::vector<Keyframe> keyframes;
    Eigen::MatrixXf matrix;
    {
        std::lock_guard<std::mutex> lock(vpr_mutex_);
        keyframes = vpr_keyframes_;
        matrix = keyframe_vpr_matrix_;
    }
    const int k = int(keyframes.size());
    if(k == 0)
        return;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "[VPR] keyframe similarity matrix (cosine): " << k << " keyframes (rows/cols in insertion order, * = culled keyframe)\n";
    out << "        ";
    for(int j = 0; j < k; j++)
        out << std::setw(7) << (std::string(keyframes[j]->is_bad() ? "*" : "") + std::to_string(keyframes[j]->keyId));
    out << "\n";
    for(int i = 0; i < k; i++){
        out << std::setw(7) << (std::string(keyframes[i]->is_bad() ? "*" : "") + std::to_string(keyframes[i]->keyId)) << " ";
        for(int j = 0; j < k; j++){
            const float d = matrix(i, j);
            if(std::isnan(d)) out << std::setw(7) << "nan";
            else out << std::setw(7) << d;
        }
        out << "\n";
    }
    std::cout << out.str() << std::flush;
}

// Cumulative per-stage timing histograms, printed through the AF_PROFILE sink.
void LocalMapping::log_profile()
{
#ifdef PROFILING_EXHAUSTIVE
    AF_PROFILE_BEGIN("Local Mapping Profiling");
    AF_PROFILE_FIELD(create_new_map_points_times_, "  Create New Map Points");
    AF_PROFILE_FIELD(search_in_neighbors_times_,   "  Search in Neighbors");
    AF_PROFILE_FIELD(local_ba_times_,              "  Local Bundle Adjustment");
    AF_PROFILE_FIELD(local_mapping_times_,         "Local Mapping");
    AF_PROFILE_END();
#endif
}

} // namespace AF_VSLAM
