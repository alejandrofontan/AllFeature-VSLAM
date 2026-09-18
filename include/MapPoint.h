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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "Observation.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace AF_VSLAM
{

class KeyFrame;
class Map;
class Frame;
class Observation;

class MapPoint;
typedef shared_ptr<AF_VSLAM::MapPoint> Pt;


class MapPoint: public std::enable_shared_from_this<MapPoint>
{
public:
    MapPoint(const vec3f &XYZ_, Keyframe pRefKF, shared_ptr<Map> pMap, const FeatureType& featureType,
             const cv::Vec3b& color = cv::Vec3b(0, 0, 0));
    std::shared_ptr<MapPoint> thisPt() {
        return shared_from_this();
    }

    void set_world_pos(const vec3f &XYZ_);
    vec3f get_world_pos();

    vec3f get_normal();
    Keyframe GetReferenceKeyFrame();

    std::map<KeyframeId,shared_ptr<Observation>> get_observations();

    // Two observation counts. number_of_observations() is the WEIGHTED count the quality
    // gates use (map-point culling, KeyFrame::tracked_map_points): an observation with
    // sensor depth (RGB-D, inv_depth > 0) counts twice, see increasePointObservability.
    // num_observing_keyframes() is the plain number of keyframes observing the point
    // (EraseObservation's discard rule, BA edge sizing).
    int number_of_observations();
    int num_observing_keyframes();
    void increasePointObservability(Keyframe projKeyframe, const KeypointIndex& projIndex);
    void decreasePointObservability(Keyframe projKeyframe, const KeypointIndex& projIndex);

    void add_observation(Keyframe projKeyframe, const KeypointIndex& projIndex);
    void EraseObservation(Keyframe projKeyframe);

    int GetIndexInKeyFrame(Keyframe pKF);
    bool is_in_keyframe(Keyframe keyframe);

    void set_bad_flag();
    bool is_bad();

    void replace(Pt pMP);
    Pt get_replaced();

    void increase_visible(int n=1);
    void increase_found(int n=1);
    float get_found_ratio();

    Pt ComputeDistinctiveDescriptors();

    cv::Mat get_descriptor();

    void UpdateNormalAndDepth();

    float get_min_distance_invariance();
    float get_max_distance_invariance();

    float PredictSize(const float &currentDist);
    float PredictSigma(const float &currentDist);

public:
    PtId ptId;

    static long unsigned int nNextId;
    long int mnFirstKFid;
    int nObs;

    // Variables used by the tracking. Default-initialized: they are scratch written by
    // Frame::is_in_frustum, but get_overlap() reads track_proj_x/y for matched points
    // that were never frustum-projected — without initializers that was an
    // uninitialized-memory read, and the garbage (recycled heap contents, different
    // every run) made the overlap-based keyframe decision nondeterministic (issue #16).
    // -1 keeps never-projected points deterministically outside get_overlap's bounds check.
    float track_proj_x{-1.0f};
    float track_proj_y{-1.0f};
    bool mbTrackInView{false};

    FrameId idLastFrameSeen{0};

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    vec3f PosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;
    FeatureType featureType;

    // Image color (OpenCV-native BGR) of the reference keypoint at creation. Fixed for the
    // point's lifetime (never averaged or re-sampled), so it is read without a lock: viewer
    // "Point Color: rgb" mode and the PLY export.
    const cv::Vec3b color;

protected:

     // Position in absolute coordinates
     vec3f XYZ;

     // Keyframes observing the point and associated observation
     std::map<KeyframeId,shared_ptr<Observation>> observations;

     // Mean viewing direction
     vec3f normalVector;

     // Best descriptor to fast matching
     cv::Mat mDescriptor;

     // Distance, keypoint size and sigma at the reference keyframe (UpdateNormalAndDepth);
     // PredictSize/PredictSigma scale them to the current viewing distance
     float refDistance;
     float refSize;
     float refSigma;
     float minDistance;
     float maxDistance;

     // refSize is a constant, not the reference keypoint's own size as in stock ORB-SLAM2
     // (kept deliberately, 2026-09-18: the size-independent projection-search radius is the
     // tuned behaviour; re-evaluate with an A/B run before changing it)
     static constexpr float reference_keypoint_size{1.5f};

     // Reference keyframe: the creating keyframe, re-pointed to another observer when it
     // stops observing the point (EraseObservation). The single source for the point's
     // scale/normal reference and for the loop-closing / GBA corrections.
     Keyframe mpRefKF;

     // Tracking counters
     int mnVisible;
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     Pt mpReplaced;

     shared_ptr<Map> mpMap;

     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
