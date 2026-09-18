/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapPoint.h"
#include "FeatureMatcher.h"

#include<mutex>

namespace AF_VSLAM
{

long unsigned int MapPoint::next_id=0;
mutex MapPoint::global_mutex;

MapPoint::MapPoint(const vec3f &XYZ_, Keyframe pRefKF, shared_ptr<Map> pMap, const FeatureType& featureType,
                   const cv::Vec3b& color):
    first_keyframe_id(pRefKF->keyId), num_observations_(0),
    last_frame_seen(0), ba_local_for_keyframe(0), fuse_candidate_for_keyframe(0), loop_point_for_keyframe(0), corrected_by_keyframe(0),
    corrected_reference(0), ba_global_for_keyframe(0), featureType(featureType), color(color), min_distance_(0), max_distance_(0),
    ref_keyframe_(pRefKF), num_visible_(1), num_found_(1), bad_(false), replaced_(static_cast<Pt>(NULL)), map_(pMap)
{

    position_ = XYZ_;
    normal_ = vec3f::Zero();

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(map_->mMutexPointCreation);
    ptId = next_id++;
}

void MapPoint::set_world_pos(const vec3f &XYZ_)
{
    unique_lock<mutex> lock2(global_mutex);
    unique_lock<mutex> lock(position_mutex_);
    position_ = XYZ_;
}

vec3f MapPoint::get_world_pos() const
{
    unique_lock<mutex> lock(position_mutex_);
    return position_;
}

vec3f MapPoint::get_normal() const
{
    unique_lock<mutex> lock(position_mutex_);
    return normal_;
}

Keyframe MapPoint::get_reference_keyframe() const
{
    unique_lock<mutex> lock(features_mutex_);
    return ref_keyframe_;
}

void MapPoint::add_observation(const Keyframe& projKeyframe, const KeypointIndex& projIndex)
{
    {
        unique_lock<mutex> lock(features_mutex_);
        if(observations_.count(projKeyframe->keyId))
            return;

        observations_[projKeyframe->keyId] = make_shared<Observation>(projKeyframe, projIndex);
        increase_observability(projKeyframe,projIndex);
    }
    compute_distinctive_descriptors()->update_normal_and_depth();
}

int MapPoint::num_observing_keyframes() const
{
    unique_lock<mutex> lock(features_mutex_);
    return int(observations_.size());
}

// An observation with sensor depth (RGB-D, inv_depth > 0) counts twice in num_observations_: it
// constrains the point's position, not only its bearing (ORB-SLAM2's stereo/RGB-D rule).
void MapPoint::increase_observability(const Keyframe& projKeyframe, const KeypointIndex& projIndex){
    if(projKeyframe->inv_depth.at(featureType)[projIndex] > 0.0f)
        num_observations_ += 2;
    else
        num_observations_++;
}

void MapPoint::decrease_observability(const Keyframe& projKeyframe, const KeypointIndex& projIndex){
    if(projKeyframe->inv_depth.at(featureType)[projIndex] > 0.0f)
        num_observations_-=2;
    else
        num_observations_--;
}
void MapPoint::erase_observation(const Keyframe& projKeyframe)
{
    bool removePoint = false;
    {
        unique_lock<mutex> lock(features_mutex_);
        if(observations_.count(projKeyframe->keyId))
        {
            KeypointIndex projIndex = observations_[projKeyframe->keyId]->projIndex;
            decrease_observability(projKeyframe,projIndex);

            observations_.erase(projKeyframe->keyId);

            // Re-point the reference keyframe only if any observation remains: erasing the
            // last observation (routine for single-observation depth-seeded points when their
            // keyframe is culled) used to dereference observations_.begin() on an EMPTY map —
            // copying a shared_ptr out of garbage memory, i.e. a refcount increment through a
            // wild pointer. That was the source of non-deterministic heap corruption and the
            // GPF segfaults inside erase_observation itself. When empty, ref_keyframe_ is left as-is:
            // the point is discarded via set_bad_flag below (size 0 <= 2) and never used again.
            if(ref_keyframe_->keyId == projKeyframe->keyId && !observations_.empty())
                ref_keyframe_ = observations_.begin()->second->projKeyframe;

            // If only 2 observations_ or less, discard point (features_mutex_ is held: read the
            // size directly, num_observing_keyframes() would re-lock)
            removePoint = (observations_.size() <= 2);
        }
    }

    if(removePoint)
        set_bad_flag();
    else
        compute_distinctive_descriptors()->update_normal_and_depth();
}

map<KeyframeId, Obs> MapPoint::get_observations() const
{
    unique_lock<mutex> lock(features_mutex_);
    return observations_;
}

int MapPoint::number_of_observations() const
{
    unique_lock<mutex> lock(features_mutex_);
    return num_observations_;
}

void MapPoint::set_bad_flag()
{
    map<KeyframeId,Obs> observations_tmp;
    {
        unique_lock<mutex> lock1(features_mutex_);
        unique_lock<mutex> lock2(position_mutex_);
        bad_ = true;
        observations_tmp = observations_;
        observations_.clear();
    }
    for(auto& obs: observations_tmp)
    {
        Keyframe keyframe = obs.second->projKeyframe;
        keyframe->erase_map_point_match(obs.second->projIndex, featureType);
    }

    map_->EraseMapPoint(this_point());
}

Pt MapPoint::get_replaced() const
{
    unique_lock<mutex> lock1(features_mutex_);
    unique_lock<mutex> lock2(position_mutex_);
    return replaced_;
}

void MapPoint::replace(const Pt& pMP)
{
    if(pMP->ptId == this->ptId)
        return;

    int nvisible, nfound;

    map<KeyframeId,Obs> observations_tmp;
    {
        unique_lock<mutex> lock1(features_mutex_);
        unique_lock<mutex> lock2(position_mutex_);
        observations_tmp = observations_;
        observations_.clear();
        bad_ = true;
        nvisible = num_visible_;
        nfound = num_found_;
        replaced_ = pMP;
    }

    for(auto& obs: observations_tmp)
    {
        // Replace measurement in keyframe
        Keyframe keyframe = obs.second->projKeyframe;

        if(!pMP->is_in_keyframe(keyframe))
        {
            keyframe->replace_map_point_match(obs.second->projIndex, pMP);
            pMP->add_observation(keyframe,obs.second->projIndex);
        }
        else
        {
            keyframe->erase_map_point_match(obs.second->projIndex, featureType);
        }
    }
    pMP->increase_found(nfound);
    pMP->increase_visible(nvisible);
    pMP->compute_distinctive_descriptors();

    map_->EraseMapPoint(this_point());
}

bool MapPoint::is_bad() const
{
    unique_lock<mutex> lock(features_mutex_);
    unique_lock<mutex> lock2(position_mutex_);
    return bad_;
}

void MapPoint::increase_visible(int n)
{
    unique_lock<mutex> lock(features_mutex_);
    num_visible_+=n;
}

void MapPoint::increase_found(int n)
{
    unique_lock<mutex> lock(features_mutex_);
    num_found_+=n;
}

float MapPoint::get_found_ratio() const
{
    unique_lock<mutex> lock(features_mutex_);
    return static_cast<float>(num_found_)/num_visible_;
}

Pt MapPoint::compute_distinctive_descriptors()
{
    // Retrieve all observed descriptors
    map<KeyframeId, Obs> observations_tmp;
    {
        unique_lock<mutex> lock1(features_mutex_);
        if(bad_)
            return this_point();
    }

    observations_tmp = get_observations();
    if(observations_tmp.empty())
        return this_point();

    vector<cv::Mat> descriptors;
    descriptors.reserve(observations_tmp.size());
    for(auto& obs: observations_tmp)
    {
        Keyframe projKeyframe = obs.second->projKeyframe;
        if(!projKeyframe->is_bad())
            descriptors.push_back(projKeyframe->descriptors.at(featureType).row(obs.second->projIndex));
    }

    if(descriptors.empty())
        return this_point();

    // Compute distances between them
    const size_t N = descriptors.size();

    std::vector<std::vector<Descriptor_Distance_Type>> Distances(N, std::vector<Descriptor_Distance_Type>(N));
    for(size_t i = 0;i < N; i++)
    {
        Distances[i][i] = Descriptor_Distance_Type(0.0);
        for(size_t j = i + 1;j < N; j++)
        {
            Descriptor_Distance_Type distij = FeatureMatcher::descriptor_distance(descriptors[i], descriptors[j], featureType);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    Descriptor_Distance_Type BestMedian = std::numeric_limits<Descriptor_Distance_Type>::max();
    int BestIdx{0};
    for(size_t i = 0;i < N; i++)
    {
        vector<Descriptor_Distance_Type> vDists(Distances[i]);
        sort(vDists.begin(),vDists.end());
        Descriptor_Distance_Type median = vDists[0.5*(N-1)];

        if(median < BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(features_mutex_);
        descriptor_ = descriptors[BestIdx].clone();
    }
    return this_point();
}

cv::Mat MapPoint::get_descriptor() const
{
    unique_lock<mutex> lock(features_mutex_);
    // Shared header, not a clone: descriptor_ is only ever REBOUND under this mutex
    // (ctor fills it once; compute_distinctive_descriptors assigns a fresh clone), never
    // written in place — so an outstanding shared view stays valid and immutable.
    return descriptor_;
}

int MapPoint::get_index_in_keyframe(const Keyframe& keyframe) const
{
    unique_lock<mutex> lock(features_mutex_);
    const auto it = observations_.find(keyframe->keyId);
    return it != observations_.end() ? static_cast<int>(it->second->projIndex) : -1;
}

bool MapPoint::is_in_keyframe(const Keyframe& keyframe) const
{
    unique_lock<mutex> lock(features_mutex_);
    return (observations_.count(keyframe->keyId));
}

void MapPoint::update_normal_and_depth()
{
    // One snapshot of observations_ AND the reference keyframe: erase_observation re-points
    // ref_keyframe_ concurrently, and indexing an older snapshot with the live reference used to
    // default-construct a null observation (operator[]) and dereference it
    map<KeyframeId , Obs> observations_tmp;
    Keyframe refKeyframe_;
    vec3f XYZ_;
    {
        unique_lock<mutex> lock1(features_mutex_);
        unique_lock<mutex> lock2(position_mutex_);
        if(bad_)
            return;

        observations_tmp = observations_;
        refKeyframe_ = ref_keyframe_;
        XYZ_ = position_;
    }
    if(observations_tmp.empty())
        return;

    const auto ref_obs = observations_tmp.find(refKeyframe_->keyId);
    if(ref_obs == observations_tmp.end())
        return;
    const KeypointIndex keyPtIdx = ref_obs->second->projIndex;

    vec3f normal{vec3f::Zero()};
    int n = 0;
    for(auto& obs: observations_tmp)
    {
        Keyframe projKeyframe = obs.second->projKeyframe;
        vec3f twc_i = projKeyframe->get_camera_center();
        vec3f normal_i = XYZ_ - twc_i;
        normal = normal + normal_i / normal_i.norm();
        n++;
    }

    vec3f PC = XYZ_ - refKeyframe_->get_camera_center();
    const float dist = PC.norm();
    const float levelScaleFactor = refKeyframe_->get_keypoint_size(keyPtIdx, featureType);

    {
        unique_lock<mutex> lock3(position_mutex_);

        ref_distance_ = dist;
        ref_size_  = reference_keypoint_size;
        ref_sigma_ = refKeyframe_->get_keypoint_sigma(keyPtIdx, featureType);

        max_distance_ = dist * levelScaleFactor;
        min_distance_ = max_distance_ / refKeyframe_->maxKeyPtSize ;

        normal_ = normal / n;
    }
}

float MapPoint::get_min_distance_invariance() const
{
    unique_lock<mutex> lock(position_mutex_);
    return 0.8f * min_distance_;
}

float MapPoint::get_max_distance_invariance() const
{
    unique_lock<mutex> lock(position_mutex_);
    return 1.2f * max_distance_;
}

float MapPoint::predict_size(const float &currentDist) const
{
    unique_lock<mutex> lock(position_mutex_);
    return ref_size_ * ref_distance_ / currentDist;
}

float MapPoint::predict_sigma(const float &currentDist) const
{
    unique_lock<mutex> lock(position_mutex_);
    return ref_sigma_ * ref_distance_ / currentDist;
}

} //namespace ORB_SLAM
