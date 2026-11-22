#include "LocalMapping.h"

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "afvslam_log.hpp"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "SettingsReader.h"

namespace AF_VSLAM
{

// ============================================================================
// Parameters
// ============================================================================

LocalMappingParameters LocalMapping::params{};

void LocalMapping::LoadParameters(const cv::FileStorage &fSettings)
{
    auto read_if_present = [&fSettings](const char* key, auto& field)
    {
        read_setting_if_present(fSettings, key, field);
    };

    std::string removed_method;
    read_if_present("LocalMapping.KeyframeCullingMethod", removed_method);
    if(!removed_method.empty() && removed_method != "information")
        AF_WARN("[LocalMapping] LocalMapping.KeyframeCullingMethod '" << removed_method << "' is ignored: the "
                << "information culler is the only one (the heuristic culler was removed)");

    read_if_present("LocalMapping.InformationKernel", params.information_kernel);
    if(params.information_kernel != "megaloc" && params.information_kernel != "covisibility")
    {
        AF_ERROR("[LocalMapping] Unknown LocalMapping.InformationKernel '" << params.information_kernel
                 << "' (options: megaloc, covisibility)");
        exit(-1);
    }
    if(params.information_kernel == "covisibility")
    {
        params.keyframe_culling_max_unexplained.store(
            LocalMappingParameters::keyframe_culling_max_unexplained_covisibility);
        params.keyframe_culling_centred = false;
    }

    read_if_present("LocalMapping.MapPointCullingMinFoundRatio", params.map_point_culling_min_found_ratio);
    read_if_present("LocalMapping.MapPointCullingMinObservations", params.map_point_culling_min_observations);
    read_if_present("LocalMapping.MapPointCullingObservationTestAge", params.map_point_culling_observation_test_age);
    read_if_present("LocalMapping.MapPointCullingProbationAge", params.map_point_culling_probation_age);
    if(params.map_point_culling_observation_test_age > params.map_point_culling_probation_age)
    {
        AF_ERROR("[LocalMapping] LocalMapping.MapPointCullingObservationTestAge ("
                 << params.map_point_culling_observation_test_age
                 << ") must not exceed LocalMapping.MapPointCullingProbationAge ("
                 << params.map_point_culling_probation_age << ")");
        exit(-1);
    }

    read_if_present("LocalMapping.CreateNewMapPointsKeyframes", params.create_new_map_points_keyframes);
    read_if_present("LocalMapping.CreateNewMapPointsMinBaselineDepthRatio",
                    params.create_new_map_points_min_baseline_depth_ratio);
    read_if_present("LocalMapping.CreateNewMapPointsMinParallaxCos", params.create_new_map_points_min_parallax_cos);
    read_if_present("LocalMapping.CreateNewMapPointsMaxParallaxCos", params.create_new_map_points_max_parallax_cos);
    if(!(params.create_new_map_points_min_parallax_cos < params.create_new_map_points_max_parallax_cos))
    {
        AF_ERROR("[LocalMapping] LocalMapping.CreateNewMapPointsMinParallaxCos ("
                 << params.create_new_map_points_min_parallax_cos
                 << ") must be below LocalMapping.CreateNewMapPointsMaxParallaxCos ("
                 << params.create_new_map_points_max_parallax_cos << ")");
        exit(-1);
    }

    read_if_present("LocalMapping.SearchInNeighborsKeyframes", params.fuse_with_neighbors_keyframes);
    read_if_present("LocalMapping.SearchInNeighborsSecondKeyframes", params.fuse_with_neighbors_second_keyframes);
    read_if_present("LocalMapping.FuseDescriptorQuantile", params.fuse_descriptor_quantile);
    if(!(params.fuse_descriptor_quantile >= 0.0f && params.fuse_descriptor_quantile <= 1.0f))
    {
        AF_ERROR("[LocalMapping] LocalMapping.FuseDescriptorQuantile must be between 0 and 1, got "
                 << params.fuse_descriptor_quantile);
        exit(-1);
    }
    read_if_present("LocalMapping.FuseEstimateMinSamples", params.fuse_estimate_min_samples);
    if(params.fuse_estimate_min_samples < 1)
    {
        AF_ERROR("[LocalMapping] LocalMapping.FuseEstimateMinSamples must be at least 1, got "
                 << params.fuse_estimate_min_samples);
        exit(-1);
    }
    // Shared with Tracking and Loop Closing
    read_if_present("Tracking.ViewingCosLimit", params.fuse_viewing_cos_limit);

    float max_unexplained = params.keyframe_culling_max_unexplained.load();
    read_if_present("LocalMapping.KeyframeCullingMaxUnexplained", max_unexplained);
    params.keyframe_culling_max_unexplained.store(max_unexplained);
    read_if_present("LocalMapping.KeyframeCullingMinAge", params.keyframe_culling_min_age);
    read_if_present("LocalMapping.KeyframeCullingMinKeyframes", params.keyframe_culling_min_keyframes);
    read_if_present("LocalMapping.KeyframeCullingScope", params.keyframe_culling_scope);
    if(params.keyframe_culling_scope != "map" && params.keyframe_culling_scope != "local")
    {
        AF_ERROR("[LocalMapping] Unknown LocalMapping.KeyframeCullingScope '" << params.keyframe_culling_scope
                 << "' (options: map, local)");
        exit(-1);
    }
    read_if_present("LocalMapping.KeyframeCullingMaxPerCall", params.keyframe_culling_max_per_call);
    read_if_present("LocalMapping.KeyframeCullingObjective", params.keyframe_culling_objective);
    if(params.keyframe_culling_objective != "unique" && params.keyframe_culling_objective != "minimax"
       && params.keyframe_culling_objective != "total-loss")
    {
        AF_ERROR("[LocalMapping] Unknown LocalMapping.KeyframeCullingObjective '" << params.keyframe_culling_objective
                 << "' (options: unique, minimax, total-loss)");
        exit(-1);
    }
    int centred = params.keyframe_culling_centred ? 1 : 0;   // cv::FileStorage has no bool reader
    read_if_present("LocalMapping.KeyframeCullingCentred", centred);
    params.keyframe_culling_centred = (centred != 0);
}

// ============================================================================
// Thread synchronization
// ============================================================================

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
    std::cout.flush();
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
    std::scoped_lock lock(stop_mutex_, finish_mutex_);
    if(finished_)
        return;

    stopped_ = false;
    stop_requested_ = false;
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

    while(is_reset_requested())
        std::this_thread::sleep_for(IDLE_POLL_PERIOD);
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

    local_mapping_times_.clear();
    last_over_budget_ = -1;

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
    std::scoped_lock lock(finish_mutex_, stop_mutex_);
    finished_ = true;
    stopped_ = true;
}

bool LocalMapping::is_finished() const
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    return finished_;
}

// ============================================================================
// Shutdown
// ============================================================================

void LocalMapping::release_map_references()
{
    current_keyframe_.reset();
    recent_map_points_.clear();
    std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
    new_keyframes_.clear();
}

}
