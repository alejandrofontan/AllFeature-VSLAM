#include "MapPoint.h"
#include "FeatureMatcher.h"
#include "Map.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace AF_VSLAM
{

// ============================================================================
// Construction
// ============================================================================

MapPoint::MapPoint(const vec3f& world_pos, Keyframe ref_keyframe, std::shared_ptr<Map> map,
                   const FeatureType& featureType, const cv::Vec3b& color):
    first_keyframe_id(ref_keyframe->keyId), featureType(featureType), color(color),
    ba_local_for_keyframe(0), fuse_candidate_for_keyframe(0), loop_point_for_keyframe(0),
    corrected_by_keyframe(0), corrected_reference(0), ba_global_for_keyframe(0),
    position_(world_pos),
    num_observations_(0), ref_keyframe_(ref_keyframe),
    normal_(vec3f::Zero()),
    num_visible_(1), num_found_(1), bad_(false), replaced_{}, map_(map)
{
    std::unique_lock<std::mutex> lock(map_->mMutexPointCreation);
    ptId = next_id++;
}

void MapPoint::LoadParameters(const cv::FileStorage &fSettings)
{
    const cv::FileNode ratio = fSettings["MapPoint.DepthInvarianceRatio"];
    if(!ratio.empty())
        ratio >> depth_invariance_ratio;

    const cv::FileNode weight = fSettings["MapPoint.DepthObservationWeight"];
    if(!weight.empty())
        weight >> depth_observation_weight;

    const cv::FileNode min_keyframes = fSettings["MapPoint.MinObservingKeyframes"];
    if(!min_keyframes.empty())
        min_keyframes >> min_observing_keyframes;
}

// ============================================================================
// Position
// ============================================================================

void MapPoint::set_world_pos(const vec3f& world_pos)
{
    std::unique_lock<std::mutex> lock(position_mutex_);
    position_ = world_pos;
}

vec3f MapPoint::get_world_pos() const
{
    std::unique_lock<std::mutex> lock(position_mutex_);
    return position_;
}

// ============================================================================
// Observations
// ============================================================================

void MapPoint::add_observation(const Keyframe& projKeyframe, const KeypointIndex& projIndex)
{
    add_observations({{projKeyframe, projIndex}});
}

void MapPoint::add_observations(const std::vector<std::pair<Keyframe, KeypointIndex>>& new_observations)
{
    bool inserted = false;
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        for(const auto& [projKeyframe, projIndex] : new_observations)
        {
            if(observations_.count(projKeyframe->keyId))
                continue;

            observations_[projKeyframe->keyId] = std::make_shared<Observation>(projKeyframe, projIndex);
            increase_observability(projKeyframe, projIndex);
            inserted = true;
        }
    }
    if(inserted)
        compute_distinctive_descriptors()->update_normal_and_depth();
}

void MapPoint::erase_observation(const Keyframe& projKeyframe)
{
    bool removePoint = false;
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        if(observations_.count(projKeyframe->keyId))
        {
            KeypointIndex projIndex = observations_[projKeyframe->keyId]->projIndex;
            decrease_observability(projKeyframe, projIndex);

            observations_.erase(projKeyframe->keyId);

            if(ref_keyframe_->keyId == projKeyframe->keyId && !observations_.empty())
                ref_keyframe_ = observations_.begin()->second->projKeyframe;

            removePoint = (static_cast<int>(observations_.size()) <= min_observing_keyframes);
        }
    }

    if(removePoint)
        set_bad_flag();
    else
        compute_distinctive_descriptors()->update_normal_and_depth();
}

int MapPoint::observation_weight(const Keyframe& projKeyframe, const KeypointIndex& projIndex) const
{
    return projKeyframe->features.at(featureType).inv_depth[projIndex] > 0.0f
               ? depth_observation_weight
               : 1;
}

void MapPoint::increase_observability(const Keyframe& projKeyframe, const KeypointIndex& projIndex)
{
    num_observations_ += observation_weight(projKeyframe, projIndex);
}

void MapPoint::decrease_observability(const Keyframe& projKeyframe, const KeypointIndex& projIndex)
{
    num_observations_ -= observation_weight(projKeyframe, projIndex);
}

std::map<KeyframeId, Obs> MapPoint::get_observations() const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return observations_;
}

std::vector<std::pair<KeyframeId, Keyframe>> MapPoint::get_observers() const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    std::vector<std::pair<KeyframeId, Keyframe>> observers;
    observers.reserve(observations_.size());
    for(const auto& [key_id, observation] : observations_)
        observers.emplace_back(key_id, observation->projKeyframe);
    return observers;
}

Keyframe MapPoint::get_reference_keyframe() const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return ref_keyframe_;
}

int MapPoint::number_of_observations() const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return num_observations_;
}

int MapPoint::num_observing_keyframes() const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return int(observations_.size());
}

int MapPoint::get_index_in_keyframe(const Keyframe& keyframe) const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    const auto it = observations_.find(keyframe->keyId);
    return it != observations_.end() ? static_cast<int>(it->second->projIndex) : -1;
}

bool MapPoint::is_in_keyframe(const Keyframe& keyframe) const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return (observations_.count(keyframe->keyId));
}

// ============================================================================
// Descriptor
// ============================================================================

Pt MapPoint::compute_distinctive_descriptors()
{
    std::map<KeyframeId, Obs> observations;
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        if(bad_)
            return this_point();

        observations = observations_;
    }

    if(observations.empty())
        return this_point();

    std::vector<cv::Mat> descriptors;
    descriptors.reserve(observations.size());
    for(auto& obs: observations)
    {
        Keyframe projKeyframe = obs.second->projKeyframe;
        if(!projKeyframe->is_bad())
            descriptors.push_back(projKeyframe->features.at(featureType).descriptors.row(obs.second->projIndex));
    }

    if(descriptors.empty())
        return this_point();

    // Compute distances between them
    const size_t num_descriptors = descriptors.size();

    std::vector<std::vector<Descriptor_Distance_Type>> distances(
        num_descriptors, std::vector<Descriptor_Distance_Type>(num_descriptors));
    for(size_t i = 0; i < num_descriptors; i++)
    {
        distances[i][i] = Descriptor_Distance_Type(0.0);
        for(size_t j = i + 1; j < num_descriptors; j++)
        {
            Descriptor_Distance_Type distance_ij =
                FeatureMatcher::descriptor_distance(descriptors[i], descriptors[j], featureType);
            distances[i][j] = distance_ij;
            distances[j][i] = distance_ij;
        }
    }

    // Take the descriptor with least median distance to the rest
    Descriptor_Distance_Type best_median = std::numeric_limits<Descriptor_Distance_Type>::max();
    size_t best_index{0};
    for(size_t i = 0; i < num_descriptors; i++)
    {
        std::vector<Descriptor_Distance_Type> distances_to_others(distances[i]);
        std::sort(distances_to_others.begin(), distances_to_others.end());
        Descriptor_Distance_Type median = distances_to_others[(num_descriptors - 1) / 2];

        if(median < best_median)
        {
            best_median = median;
            best_index = i;
        }
    }

    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        descriptor_ = descriptors[best_index].clone();
    }
    return this_point();
}

cv::Mat MapPoint::get_descriptor() const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return descriptor_;
}

// ============================================================================
// Normal, scale and depth
// ============================================================================

void MapPoint::update_normal_and_depth()
{
    std::map<KeyframeId, Obs> observations;
    Keyframe ref_keyframe;
    vec3f world_pos;
    {
        std::unique_lock<std::mutex> lock1(features_mutex_);
        std::unique_lock<std::mutex> lock2(position_mutex_);
        if(bad_)
            return;

        observations = observations_;
        ref_keyframe = ref_keyframe_;
        world_pos = position_;
    }
    if(observations.empty())
        return;

    if(observations.find(ref_keyframe->keyId) == observations.end())
        return;

    vec3f normal{vec3f::Zero()};
    int num_observers = 0;
    for(auto& obs: observations)
    {
        Keyframe projKeyframe = obs.second->projKeyframe;
        vec3f camera_centre = projKeyframe->get_twc();
        vec3f viewing_direction = world_pos - camera_centre;
        normal = normal + viewing_direction / viewing_direction.norm();
        num_observers++;
    }

    vec3f ref_to_point = world_pos - ref_keyframe->get_twc();
    const float dist = ref_to_point.norm();

    {
        std::unique_lock<std::mutex> lock3(position_mutex_);

        ref_distance_ = dist;
        ref_size_  = reference_keypoint_size;

        max_distance_ = dist * depth_invariance_ratio;
        min_distance_ = dist / depth_invariance_ratio;

        normal_ = normal / num_observers;
    }
}

vec3f MapPoint::get_normal() const
{
    std::unique_lock<std::mutex> lock(position_mutex_);
    return normal_;
}

float MapPoint::get_min_distance_invariance() const
{
    std::unique_lock<std::mutex> lock(position_mutex_);
    return min_distance_;
}

float MapPoint::get_max_distance_invariance() const
{
    std::unique_lock<std::mutex> lock(position_mutex_);
    return max_distance_;
}

float MapPoint::predict_size(const float &currentDist) const
{
    std::unique_lock<std::mutex> lock(position_mutex_);
    return ref_size_ * ref_distance_ / currentDist;
}

// ============================================================================
// Tracking statistics
// ============================================================================

void MapPoint::increase_visible(int n)
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    num_visible_ += n;
}

void MapPoint::increase_found(int n)
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    num_found_ += n;
}

float MapPoint::get_found_ratio() const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return static_cast<float>(num_found_)/num_visible_;
}

// ============================================================================
// Lifecycle
// ============================================================================

void MapPoint::set_bad_flag()
{
    std::map<KeyframeId, Obs> observations;
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        bad_ = true;
        observations = observations_;
        observations_.clear();
    }
    for(auto& obs: observations)
    {
        Keyframe keyframe = obs.second->projKeyframe;
        keyframe->erase_map_point_match(this_point(), obs.second->projIndex);
    }

    map_->EraseMapPoint(this_point());
}

bool MapPoint::is_bad() const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return bad_;
}

void MapPoint::replace(const Pt& replacement)
{
    if(replacement->ptId == this->ptId)
        return;

    int num_visible, num_found;

    std::map<KeyframeId, Obs> observations;
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        observations = observations_;
        observations_.clear();
        bad_ = true;
        num_visible = num_visible_;
        num_found = num_found_;
        replaced_ = replacement;
    }

    std::vector<std::pair<Keyframe, KeypointIndex>> merged;
    merged.reserve(observations.size());
    for(auto& obs: observations)
    {
        Keyframe keyframe = obs.second->projKeyframe;

        if(!replacement->is_in_keyframe(keyframe))
        {
            if(keyframe->replace_map_point_match(this_point(), obs.second->projIndex, replacement))
                merged.emplace_back(keyframe, obs.second->projIndex);
        }
        else
        {
            keyframe->erase_map_point_match(this_point(), obs.second->projIndex);
        }
    }
    replacement->increase_found(num_found);
    replacement->increase_visible(num_visible);
    replacement->add_observations(merged);

    map_->EraseMapPoint(this_point());
}

Pt MapPoint::get_replaced() const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return replaced_;
}

}
