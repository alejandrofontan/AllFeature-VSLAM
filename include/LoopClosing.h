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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "MapDrawer.h"
#include "FeatureMatcher.h"

#include "PlaceRecognition.h"

#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>

#include "g2o/types/types_seven_dof_expmap.h"

namespace AF_VSLAM
{

class LocalMapping;
class MapDrawer;

// Same pattern as TrackingParameters/LocalMappingParameters: compiled-in defaults,
// overridden only by keys present in the settings file, so settings YAMLs without a
// LoopClosing.* block keep working unchanged.
struct LoopClosingParameters
{
    // detect_loop()
    int covisibility_consistency_threshold{3}; // a candidate's covisibility group must be seen in this many consecutive keyframes
    int min_keyframes_between_loops{10};       // no loop detection for this many keyframes after a loop closure (and at the start of a map)
};

// A loop candidate's covisibility group (its id plus its connected keyframes' ids,
// sorted) and the number of consecutive keyframes whose candidates hit it.
struct ConsistentGroup
{
    std::vector<KeyframeId> keyframes;
    int consistency{0};
};

class LoopConnections{
public:
    KeyframeId keyframeId{};
    Keyframe keyframe{};
    std::map<KeyframeId,Keyframe> connections{};
    LoopConnections() = default;
    LoopConnections(const KeyframeId & keyframeId, const Keyframe& keyframe, const std::map<KeyframeId,Keyframe>& connections);
};

class LoopClosing
{
public:

    typedef map<Keyframe ,g2o::Sim3,std::less<Keyframe>,
            Eigen::aligned_allocator<std::pair<Keyframe const, g2o::Sim3> > > KeyFrameAndPose;

public:

    // featureType (geometric verification of loop candidates) is the backend's
    // verification_feature(); the backend also owns the keyframe database.
    // local_mapper is paused/released around loop corrections; map_drawer records the
    // closed loops for the viewer. Both exist before this object (System's constructor).
    LoopClosing(shared_ptr<Map> pMap, shared_ptr<PlaceRecognition> place_recognition,
        std::shared_ptr<LocalMapping> local_mapper, std::shared_ptr<MapDrawer> map_drawer,
        const bool bFixScale, const std::vector<FeatureType>& feat_types,
        int image_width, int image_height);

    // Tunable parameters, loaded from the settings YAML at System startup
    static LoopClosingParameters params;
    static void LoadParameters(const cv::FileStorage& fSettings);

    // Main function
    void Run();

    // Keyframe queue, fed by LocalMapping (process_keyframe, after each keyframe's
    // mapping iteration) and drained by Run() (detect_loop). Without an active VPR
    // backend the thread never runs (System's constructor), so insert_keyframe drops
    // the keyframe instead of queueing it forever.
    void insert_keyframe(const Keyframe& keyframe);
    bool has_new_keyframes() const;

    // Reset protocol, used by Tracking::reset: request_reset() blocks the caller until
    // the Loop Closing thread has dropped its keyframe queue and loop-detection state
    // (reset_if_requested, at the end of each Run() iteration). Returns at once when
    // the VPR backend is inactive (no thread would ever clear the request).
    void request_reset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }

    // Finish protocol, used by System::Shutdown: request_finish() makes Run() return at
    // its next check; is_finished() turns true once it has. finished_ starts true, so a
    // system without an active VPR backend (thread never started) shuts down at once.
    void request_finish();
    bool is_finished() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    vector<double> loopClosingTime{};
    size_t numOfLoopClosures{0};

protected:

    // Loop detection, one queued keyframe per call: pops it into current_keyframe_,
    // retrieves its loop candidates from the VPR backend, and keeps the ones whose
    // covisibility group has been retrieved by consecutive keyframes
    // (consistent_loop_candidates). True when loop_candidates_ is non-empty: ComputeSim3
    // then verifies them geometrically. Every keyframe joins the VPR database here.
    bool detect_loop();
    std::vector<Keyframe> consistent_loop_candidates(const std::vector<Keyframe>& candidates);

    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();

    // Called by Run(): perform a pending reset
    void reset_if_requested();
    bool is_reset_requested() const;
    mutable std::mutex reset_mutex_;   // guards reset_requested_
    bool reset_requested_{false};

    // Called by Run(): exit when asked / publish the exit
    bool is_finish_requested() const;
    void set_finished();
    mutable std::mutex finish_mutex_;   // guards finish_requested_ and finished_
    bool finish_requested_{false};
    bool finished_{true};

    FeatureType featureType;
    std::vector<FeatureType> feat_types;

    shared_ptr<Map> mpMap;
    std::shared_ptr<MapDrawer> map_drawer_;

    shared_ptr<PlaceRecognition> place_recognition;

    std::shared_ptr<LocalMapping> local_mapper_;

    mutable std::mutex new_keyframes_mutex_;   // guards new_keyframes_
    std::list<Keyframe> new_keyframes_;

    // Loop detector state (detect_loop): the keyframe being processed, the covisibility
    // groups retrieved for the previous keyframe with their consistency counts, and the
    // candidates that passed the consistency vote (input of ComputeSim3)
    Keyframe current_keyframe_;
    std::vector<ConsistentGroup> consistent_groups_;
    std::vector<Keyframe> loop_candidates_;
    KeyframeId last_loop_keyframe_id_{0};   // keyId of the last closed loop's keyframe

    Keyframe  mpMatchedKF;
    std::vector<Keyframe> mvpCurrentConnectedKFs;
    std::vector<Pt> mvpCurrentMatchedPoints;
    std::vector<Pt> mvpLoopMapPoints;
    mat4f mScw;
    g2o::Sim3 mg2oScw;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;
    bool mnFullBAIdx;

    int image_width;
    int image_height;

    std::shared_ptr<FeatureMatcher> matcher;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
