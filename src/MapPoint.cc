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

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const vec3f &XYZ_, Keyframe pRefKF, shared_ptr<Map> pMap, const FeatureType& featureType,
                   const cv::Vec3b& color):
    mnFirstKFid(pRefKF->keyId), nObs(0),
    idLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), featureType(featureType), color(color), minDistance(0), maxDistance(0),
    mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(static_cast<Pt>(NULL)), mpMap(pMap)
{

    XYZ = XYZ_;
    normalVector = vec3f::Zero();

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    ptId = nNextId++;
}

void MapPoint::set_world_pos(const vec3f &XYZ_)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    XYZ = XYZ_;
}

vec3f MapPoint::get_world_pos()
{
    unique_lock<mutex> lock(mMutexPos);
    return XYZ;
}

vec3f MapPoint::get_normal()
{
    unique_lock<mutex> lock(mMutexPos);
    return normalVector;
}

Keyframe MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapPoint::add_observation(Keyframe projKeyframe,  const KeypointIndex& projIndex)
{
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(observations.count(projKeyframe->keyId))
            return;

        observations[projKeyframe->keyId] = make_shared<Observation>(projKeyframe, projIndex);
        increasePointObservability(projKeyframe,projIndex);
    }
    ComputeDistinctiveDescriptors()->UpdateNormalAndDepth();
}

int MapPoint::num_observing_keyframes()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return int(observations.size());
}

// An observation with sensor depth (RGB-D, inv_depth > 0) counts twice in nObs: it
// constrains the point's position, not only its bearing (ORB-SLAM2's stereo/RGB-D rule).
void MapPoint::increasePointObservability(Keyframe projKeyframe, const KeypointIndex& projIndex){
    if(projKeyframe->inv_depth.at(featureType)[projIndex] > 0.0f)
        nObs += 2;
    else
        nObs++;
}

void MapPoint::decreasePointObservability(Keyframe projKeyframe, const KeypointIndex& projIndex){
    if(projKeyframe->inv_depth.at(featureType)[projIndex] > 0.0f)
        nObs-=2;
    else
        nObs--;
}
void MapPoint::EraseObservation(Keyframe projKeyframe)
{
    bool removePoint = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(observations.count(projKeyframe->keyId))
        {
            KeypointIndex projIndex = observations[projKeyframe->keyId]->projIndex;
            decreasePointObservability(projKeyframe,projIndex);

            observations.erase(projKeyframe->keyId);

            // Re-point the reference keyframe only if any observation remains: erasing the
            // last observation (routine for single-observation depth-seeded points when their
            // keyframe is culled) used to dereference observations.begin() on an EMPTY map —
            // copying a shared_ptr out of garbage memory, i.e. a refcount increment through a
            // wild pointer. That was the source of non-deterministic heap corruption and the
            // GPF segfaults inside EraseObservation itself. When empty, mpRefKF is left as-is:
            // the point is discarded via set_bad_flag below (size 0 <= 2) and never used again.
            if(mpRefKF->keyId == projKeyframe->keyId && !observations.empty())
                mpRefKF = observations.begin()->second->projKeyframe;

            // If only 2 observations or less, discard point (mMutexFeatures is held: read the
            // size directly, num_observing_keyframes() would re-lock)
            removePoint = (observations.size() <= 2);
        }
    }

    if(removePoint)
        set_bad_flag();
    else
        ComputeDistinctiveDescriptors()->UpdateNormalAndDepth();
}

map<KeyframeId, Obs> MapPoint::get_observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return observations;
}

int MapPoint::number_of_observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapPoint::set_bad_flag()
{
    map<KeyframeId,Obs> observations_tmp;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        observations_tmp = observations;
        observations.clear();
    }
    for(auto& obs: observations_tmp)
    {
        Keyframe keyframe = obs.second->projKeyframe;
        keyframe->EraseMapPointMatch(obs.second->projIndex, featureType);
    }

    mpMap->EraseMapPoint(thisPt());
}

Pt MapPoint::get_replaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::replace(Pt pMP)
{
    if(pMP->ptId == this->ptId)
        return;

    int nvisible, nfound;

    map<KeyframeId,Obs> observations_tmp;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        observations_tmp = observations;
        observations.clear();
        mbBad = true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for(auto& obs: observations_tmp)
    {
        // Replace measurement in keyframe
        Keyframe keyframe = obs.second->projKeyframe;

        if(!pMP->is_in_keyframe(keyframe))
        {
            keyframe->ReplaceMapPointMatch(obs.second->projIndex, pMP);
            pMP->add_observation(keyframe,obs.second->projIndex);
        }
        else
        {
            keyframe->EraseMapPointMatch(obs.second->projIndex, featureType);
        }
    }
    pMP->increase_found(nfound);
    pMP->increase_visible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(thisPt());
}

bool MapPoint::is_bad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::increase_visible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::increase_found(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::get_found_ratio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

Pt MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    map<KeyframeId, Obs> observations_tmp;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return thisPt();
    }

    observations_tmp = get_observations();
    if(observations_tmp.empty())
        return thisPt();

    vector<cv::Mat> descriptors;
    descriptors.reserve(observations_tmp.size());
    for(auto& obs: observations_tmp)
    {
        Keyframe projKeyframe = obs.second->projKeyframe;
        if(!projKeyframe->is_bad())
            descriptors.push_back(projKeyframe->descriptors.at(featureType).row(obs.second->projIndex));
    }

    if(descriptors.empty())
        return thisPt();

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
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = descriptors[BestIdx].clone();
    }
    return thisPt();
}

cv::Mat MapPoint::get_descriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    // Shared header, not a clone: mDescriptor is only ever REBOUND under this mutex
    // (ctor fills it once; ComputeDistinctiveDescriptors assigns a fresh clone), never
    // written in place — so an outstanding shared view stays valid and immutable.
    return mDescriptor;
}

int MapPoint::GetIndexInKeyFrame(Keyframe keyframe)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(observations.count(keyframe->keyId))
        return observations[keyframe->keyId]->projIndex;
    else
        return -1;
}

bool MapPoint::is_in_keyframe(Keyframe keyframe)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (observations.count(keyframe->keyId));
}

void MapPoint::UpdateNormalAndDepth()
{
    // One snapshot of observations AND the reference keyframe: EraseObservation re-points
    // mpRefKF concurrently, and indexing an older snapshot with the live reference used to
    // default-construct a null observation (operator[]) and dereference it
    map<KeyframeId , Obs> observations_tmp;
    Keyframe refKeyframe_;
    vec3f XYZ_;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;

        observations_tmp = observations;
        refKeyframe_ = mpRefKF;
        XYZ_ = XYZ;
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
    const float levelScaleFactor = refKeyframe_->GetKeyPtSize(keyPtIdx, featureType);

    {
        unique_lock<mutex> lock3(mMutexPos);

        refDistance = dist;
        refSize  = reference_keypoint_size;
        refSigma = refKeyframe_->GetKeyPt1DSigma(keyPtIdx, featureType);

        maxDistance = dist * levelScaleFactor;
        minDistance = maxDistance / refKeyframe_->maxKeyPtSize ;

        normalVector = normal / n;
    }
}

float MapPoint::get_min_distance_invariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * minDistance;
}

float MapPoint::get_max_distance_invariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * maxDistance;
}

float MapPoint::PredictSize(const float &currentDist)
{
    unique_lock<mutex> lock(mMutexPos);
    return refSize * refDistance / currentDist;
}

float MapPoint::PredictSigma(const float &currentDist)
{
    unique_lock<mutex> lock(mMutexPos);
    return refSigma * refDistance / currentDist;
}

} //namespace ORB_SLAM
