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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "FeatureExtractor.h"
#include "Frame.h"
#include "PlaceRecognition.h"
#include "FeatureFactory.h"

#include <mutex>


namespace AF_VSLAM
{
// Legacy ORB-SLAM2 headers use unqualified std types throughout; this using-directive
// previously arrived transitively through the (now removed) DBoW2 vocabulary headers.
using namespace std;

class Map;
class Frame;

class KeyFrame;
typedef shared_ptr<AF_VSLAM::KeyFrame> Keyframe;
class MapPoint;
typedef shared_ptr<AF_VSLAM::MapPoint> Pt;

// Keyframe set ordered by keyId instead of the shared_ptr's pointer value: pointer
// order differs between runs (allocator/ASLR), which made iteration order — and
// every tie broken by it (spanning-tree reassignment, local-map expansion) —
// nondeterministic (issue #16). Defined out-of-line below the class (needs keyId).
struct KeyframeIdLess
{
    bool operator()(const Keyframe& a, const Keyframe& b) const;
};
using KeyframeIdSet = std::set<Keyframe, KeyframeIdLess>;

class KeyFrame : public std::enable_shared_from_this<KeyFrame>
{
public:
    KeyFrame(Frame &F, shared_ptr<Map> pMap, shared_ptr<PlaceRecognition> place_recognition);
    std::shared_ptr<KeyFrame> thisKeyframe() {
        return shared_from_this();
    }

    // Pose functions
    void set_pose(const mat4f &Tcw_);
    mat4f get_pose() const;
    mat4f get_pose_inverse() const;
    vec3f get_camera_center() const;
    mat3f get_rotation() const;
    vec3f get_translation() const;
    void getFullIntrinsics(float &fx, float &fy, float &cx, float &cy, float& invfx, float& invfy) const;
    void getFullPose(mat4f &Twc_, mat3f &Rwc_, vec3f &twc_, mat4f &Tcw_, mat3f &Rcw_, vec3f &tcw_) const;

    // Compute the global descriptor with the active VPR backend (MegaLoc image
    // embedding — which then releases `image`); no-op when VPR is inactive.
    void compute_global_descriptor();

    // Similarity of the two keyframes' global descriptors through the VPR backend
    // (MegaLoc: cosine). NaN when VPR is inactive. Hook for
    // appearance-based keyframe culling.
    float vpr_similarity(const Keyframe& other) const;

    // Covisibility graph functions
    void AddConnection(const Keyframe& pKF, const int &weight);
    void EraseConnection(const Keyframe& pKF);
    void update_connections();
    void UpdateBestCovisibles();
    map<KeyframeId,Keyframe> GetConnectedKeyFrames() const;
    std::vector<Keyframe > get_covisible_keyframes() const;
    std::vector<Keyframe> get_best_covisibility_keyframes(const int &N) const;
    std::vector<Keyframe> GetCovisiblesByWeight(const int &w) const;
    int GetWeight(const Keyframe& pKF) const;

    // Spanning tree functions
    void AddChild(const Keyframe& pKF);
    void EraseChild(const Keyframe& pKF);
    void ChangeParent(const Keyframe& pKF);
    KeyframeIdSet get_children() const;
    Keyframe get_parent() const;
    bool hasChild(const Keyframe& pKF) const;

    // Loop Edges
    void AddLoopEdge(const Keyframe& pKF);
    KeyframeIdSet GetLoopEdges() const;

    // MapPoint observation functions
    Pt create_monocular_map_point(const vec3f& worldPos,
                             const KeypointIndex& refIndex,
                             const Keyframe& projKeyframe, const KeypointIndex& projIndex,
                             const FeatureType& featureType);
    Pt create_map_point(const vec3f& worldPos,
                      const KeypointIndex& refIndex,
                      const FeatureType& featureType);

    void add_map_point(const Pt& pt, const KeypointIndex& index);
    void EraseMapPointMatch(const size_t &idx, const FeatureType& featType);
    void EraseMapPointMatch(const Pt& pMP);
    void ReplaceMapPointMatch(const size_t &idx, const Pt& pMP);
    std::set<Pt> get_map_points(const FeatureType& featType) const;
    std::vector<Pt> get_map_point_matches(const FeatureType& feat_type) const;
    int tracked_map_points(const int &minObs) const;
    Pt get_map_point(const size_t &idx, const FeatureType& featType) const;

    // KeyPoint functions
    std::vector<size_t> get_features_in_area(const float &x, const float  &y, const float  &r, const FeatureType& featType) const;

    // Image
    bool is_in_image(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void set_bad_flag();
    bool is_bad() const;

    // Compute Scene Depth (q=2 median). Used in monocular. -1 when the keyframe has no map points.
    float compute_scene_median_depth(const int q) const;

    static bool weightComp( int a, int b){
        return a>b;
    }

    static bool lId(const Keyframe& pKF1, const Keyframe& pKF2){
        return pKF1->keyId < pKF2->keyId;
    }

    [[nodiscard]] float GetKeyPtSize(const KeypointIndex &keyPtIdx, const FeatureType& featType) const;
    [[nodiscard]] float GetKeyPt1DSigma2(const KeypointIndex &keyPtIdx, const FeatureType& featType) const;
    [[nodiscard]] float get_keypt_1Dinf(const KeypointIndex &keyPtIdx, const FeatureType& featType) const;
    [[nodiscard]] mat2f GetKeyPt2DInf(const KeypointIndex &keyPtIdx, const FeatureType& featType) const;
    [[nodiscard]] float GetKeyPt1DSigma(const KeypointIndex &keyPtIdx, const FeatureType& featType) const;

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    std::vector<FeatureType> featureTypes{};
    std::map<FrameId, std::vector<cv::DMatch>> cache_matched_pairs{};
    std::map<FrameId, std::map<FeatureType, std::vector<cv::DMatch>>> cache_matched_pairs_feat_type{};

    static long unsigned int nNextId;
    KeyframeId keyId;
    const long unsigned int frame_id;

    const double timestamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by loop closing
    mat4f TcwGBA;
    mat4f TcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const std::map<FeatureType, int> N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::map<FeatureType, std::vector<cv::KeyPoint>> mvKeys;
    const std::map<FeatureType, std::vector<cv::KeyPoint>> keypoints;
    const std::map<FeatureType, std::vector<float>> inv_depth; // inverse depth; 0 where no valid depth
    const std::map<FeatureType, std::vector<float>> sigma2invDepth; // variance of inv_depth; 0 where no valid depth
    const std::map<FeatureType, std::vector<cv::Vec3b>> keypoint_colors; // BGR under each keypoint (Frame::GetColors); map points created here inherit it
    std::map<FeatureType, cv::Mat> descriptors;


    // Image-embedding VPR (MegaLoc): the keyframe's image (OpenCV-native BGR, shared
    // header from the Frame) and the frame's already-computed global descriptor (if
    // Tracking embedded the frame for the keyframe-information decision) live only until
    // compute_global_descriptor() has stored the descriptor in placecell
    // (System::place_cell, keyed by frame_id) — the descriptor is preferred, the image is
    // embedded only when no descriptor came with the frame; both are released afterwards.
    // Read the stored descriptor via place_cell->descriptor(frame_id).
    cv::Mat image;
    Eigen::VectorXf global_descriptor;

    // Scale
    float sizeTolerance{};
    std::map<FeatureType, vector<mat2f>> keyPtsSigma2{};
    std::map<FeatureType, vector<mat2f>> keyPtsInf{};
    std::map<FeatureType, vector<float>> keyPtsSize{};
    float maxKeyPtSize{};
    float maxKeyPtSigma{};

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;

    const cv::Mat mK;

    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    mat4f Tcw;
    mat4f Twc;
    vec3f twc;

    // MapPoints associated to keypoints
    std::map<FeatureType, std::vector<Pt>> mvpMapPoints;

    // Visual place recognition backend (owns the keyframe database this keyframe is
    // registered in; erased from it in set_bad_flag)
    shared_ptr<PlaceRecognition> place_recognition_;

    // Grid over the image to speed up feature matching
    std::map<FeatureType, std::vector< std::vector <std::vector<size_t>>>> mGrid;

    std::map<KeyframeId, Keyframe> connectedKeyFrames;
    std::map<KeyframeId,int> connectedKeyFrameWeights;
    std::vector<Keyframe> orderedConnectedKeyFrames;
    std::vector<int> orderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    Keyframe mpParent;
    KeyframeIdSet mspChildrens;
    KeyframeIdSet mspLoopEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;

    shared_ptr<Map> mpMap;

    // mutable: pure queries lock them and stay const
    mutable std::mutex mMutexPose;
    mutable std::mutex mMutexConnections;
    mutable std::mutex mMutexFeatures;
};

inline bool KeyframeIdLess::operator()(const Keyframe& a, const Keyframe& b) const
{
    return a->keyId < b->keyId;
}

} //namespace ORB_SLAM

#endif // KEYFRAME_H
