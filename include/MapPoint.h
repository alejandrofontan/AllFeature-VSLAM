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
    std::shared_ptr<MapPoint> this_point() {
        return shared_from_this();
    }

    void set_world_pos(const vec3f &XYZ_);
    vec3f get_world_pos() const;

    vec3f get_normal() const;
    Keyframe get_reference_keyframe() const;

    std::map<KeyframeId,shared_ptr<Observation>> get_observations() const;

    // Two observation counts. number_of_observations() is the WEIGHTED count the quality
    // gates use (map-point culling, KeyFrame::tracked_map_points): an observation with
    // sensor depth (RGB-D, inv_depth > 0) counts twice, see increase_observability.
    // num_observing_keyframes() is the plain number of keyframes observing the point
    // (erase_observation's discard rule, BA edge sizing).
    int number_of_observations() const;
    int num_observing_keyframes() const;
    void increase_observability(const Keyframe& projKeyframe, const KeypointIndex& projIndex);
    void decrease_observability(const Keyframe& projKeyframe, const KeypointIndex& projIndex);

    void add_observation(const Keyframe& projKeyframe, const KeypointIndex& projIndex);
    void erase_observation(const Keyframe& projKeyframe);

    int get_index_in_keyframe(const Keyframe& pKF) const;
    bool is_in_keyframe(const Keyframe& keyframe) const;

    void set_bad_flag();
    bool is_bad() const;

    void replace(const Pt& pMP);
    Pt get_replaced() const;

    void increase_visible(int n=1);
    void increase_found(int n=1);
    float get_found_ratio() const;

    Pt compute_distinctive_descriptors();

    cv::Mat get_descriptor() const;

    void update_normal_and_depth();

    float get_min_distance_invariance() const;
    float get_max_distance_invariance() const;

    float predict_size(const float &currentDist) const;
    float predict_sigma(const float &currentDist) const;

public:
    PtId ptId;

    static long unsigned int next_id;
    long int first_keyframe_id;
    int num_observations_;

    // Variables used by the tracking. Default-initialized: they are scratch written by
    // Frame::is_in_frustum, but get_overlap() reads track_proj_x/y for matched points
    // that were never frustum-projected — without initializers that was an
    // uninitialized-memory read, and the garbage (recycled heap contents, different
    // every run) made the overlap-based keyframe decision nondeterministic (issue #16).
    // -1 keeps never-projected points deterministically outside get_overlap's bounds check.
    float track_proj_x{-1.0f};
    float track_proj_y{-1.0f};
    bool track_in_view{false};

    FrameId last_frame_seen{0};

    // Variables used by local mapping
    long unsigned int ba_local_for_keyframe;
    long unsigned int fuse_candidate_for_keyframe;

    // Variables used by loop closing
    long unsigned int loop_point_for_keyframe;
    long unsigned int corrected_by_keyframe;
    long unsigned int corrected_reference;
    vec3f position_gba;
    long unsigned int ba_global_for_keyframe;


    static std::mutex global_mutex;
    FeatureType featureType;

    // Image color (OpenCV-native BGR) of the reference keypoint at creation. Fixed for the
    // point's lifetime (never averaged or re-sampled), so it is read without a lock: viewer
    // "Point Color: rgb" mode and the PLY export.
    const cv::Vec3b color;

protected:

     // Position in absolute coordinates
     vec3f position_;

     // Keyframes observing the point and associated observation
     std::map<KeyframeId,shared_ptr<Observation>> observations_;

     // Mean viewing direction
     vec3f normal_;

     // Best descriptor to fast matching
     cv::Mat descriptor_;

     // Distance, keypoint size and sigma at the reference keyframe (update_normal_and_depth);
     // predict_size/predict_sigma scale them to the current viewing distance
     float ref_distance_;
     float ref_size_;
     float ref_sigma_;
     float min_distance_;
     float max_distance_;

     // ref_size_ is a constant, not the reference keypoint's own size as in stock ORB-SLAM2
     // (kept deliberately, 2026-09-18: the size-independent projection-search radius is the
     // tuned behaviour; re-evaluate with an A/B run before changing it)
     static constexpr float reference_keypoint_size{1.5f};

     // Reference keyframe: the creating keyframe, re-pointed to another observer when it
     // stops observing the point (erase_observation). The single source for the point's
     // scale/normal reference and for the loop-closing / GBA corrections.
     Keyframe ref_keyframe_;

     // Tracking counters
     int num_visible_;
     int num_found_;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool bad_;
     Pt replaced_;

     shared_ptr<Map> map_;

     // mutable: pure queries lock them and stay const
     mutable std::mutex position_mutex_;
     mutable std::mutex features_mutex_;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
