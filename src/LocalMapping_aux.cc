/**
 * Auxiliary LocalMapping members kept out of LocalMapping.cc so the
 * per-keyframe flow there reads uninterrupted (companion to
 * include/LocalMapping_aux.h): parameter loading, the online keyframe VPR
 * similarity matrix, and profiling output.
 */
#include "LocalMapping.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

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
