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

#ifndef AF_VSLAM_LOOP_CLOSING_H
#define AF_VSLAM_LOOP_CLOSING_H

#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "g2o/types/types_seven_dof_expmap.h"

#include "KeyFrame.h"
#include "Map.h"
#include "PlaceRecognition.h"

namespace AF_VSLAM
{

class FeatureMatcher;
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

    // compute_sim3()
    int sim3_min_matches{20};   // candidates with fewer feature matches to the current keyframe get no Sim3 solver
    int sim3_min_inliers{20};   // inliers of the optimized Sim3 that accept a candidate

    // search_loop_map_points()
    int loop_min_matches{40};   // loop-side map points matched into the current keyframe (RANSAC inliers + projection search) that accept the loop

    // search_and_fuse()
    float fuse_radius{4.0f};    // projection search radius (pixels) when fusing the loop-side points into the corrected keyframes

    // run_global_bundle_adjustment()
    int gba_iterations{10};
};

// A loop candidate's covisibility group (its id plus its connected keyframes' ids,
// sorted) and the number of consecutive keyframes whose candidates hit it.
struct ConsistentGroup
{
    std::vector<KeyframeId> keyframes;
    int consistency{0};
};

// A keyframe of the corrected side of a loop and the covisibility links it gained
// through the loop fusion: the loop edges of Optimizer::OptimizeEssentialGraph.
struct LoopConnections
{
    Keyframe keyframe{};
    std::map<KeyframeId, Keyframe> connections{};
};

// Loop Closing thread. Consumes the keyframes LocalMapping hands over after their
// mapping iteration (run/detect_loop): retrieves loop candidates from the VPR backend,
// verifies the consistent ones geometrically (compute_sim3, search_loop_map_points),
// corrects the map around the loop (correct_loop: Sim3 propagation, point fusion,
// essential-graph optimization) and launches a global bundle adjustment in its own
// thread (run_global_bundle_adjustment, apply_gba_correction).
class LoopClosing
{
public:
    // Keyframe -> Sim3 pose; aligned allocator for g2o::Sim3's fixed-size Eigen members
    using KeyframePoses = std::map<Keyframe, g2o::Sim3, std::less<Keyframe>,
                                   Eigen::aligned_allocator<std::pair<const Keyframe, g2o::Sim3>>>;

    // The VPR backend provides the loop candidates and names the local feature that
    // verifies them (verification_feature). local_mapper is paused/released around loop
    // corrections; map_drawer records the closed loops for the viewer. Both exist before
    // this object (System's constructor). fix_scale: stereo/RGB-D (SE3 correction).
    LoopClosing(std::shared_ptr<Map> map, std::shared_ptr<PlaceRecognition> place_recognition,
                std::shared_ptr<LocalMapping> local_mapper, std::shared_ptr<MapDrawer> map_drawer,
                bool fix_scale, const std::vector<FeatureType>& feature_types,
                int image_width, int image_height);
    // Joins a finished global BA thread (System::Shutdown waits for is_gba_running() first)
    ~LoopClosing();

    // Tunable parameters, loaded from the settings YAML at System startup
    static LoopClosingParameters params;
    static void LoadParameters(const cv::FileStorage& fSettings);

    // Thread body
    void run();

    // Keyframe queue, fed by LocalMapping (process_keyframe, after each keyframe's
    // mapping iteration) and drained by run() (detect_loop). Without an active VPR
    // backend the thread never runs (System's constructor), so insert_keyframe drops
    // the keyframe instead of queueing it forever.
    void insert_keyframe(const Keyframe& keyframe);
    bool has_new_keyframes() const;

    // Reset protocol, used by Tracking::reset: request_reset() blocks the caller until
    // the Loop Closing thread has dropped its keyframe queue and loop-detection state
    // (reset_if_requested, at the end of each run() iteration). Returns at once when
    // the VPR backend is inactive (no thread would ever clear the request).
    void request_reset();

    // True while a global bundle adjustment thread is running (System::Shutdown waits
    // for it to clear before joining the threads)
    bool is_gba_running() const;

    // Finish protocol, used by System::Shutdown: request_finish() makes run() return at
    // its next check; is_finished() turns true once it has. finished_ starts true, so a
    // system without an active VPR backend (thread never started) shuts down at once.
    void request_finish();
    bool is_finished() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    // Loop detection, one queued keyframe per call: pops it into current_keyframe_,
    // retrieves its loop candidates from the VPR backend, and keeps the ones whose
    // covisibility group has been retrieved by consecutive keyframes
    // (consistent_loop_candidates). True when loop_candidates_ is non-empty. Every
    // keyframe joins the VPR database here.
    bool detect_loop();
    std::vector<Keyframe> consistent_loop_candidates(const std::vector<Keyframe>& candidates);

    // Geometric verification of loop_candidates_: feature matches to the current keyframe,
    // alternating Sim3 RANSAC over the candidates, Sim3 optimization. On success
    // matched_keyframe_, g2o_Scw_/Scw_ and loop_matched_points_ are set;
    // search_loop_map_points then projects the matched keyframe's neighbourhood into the
    // current keyframe with that Sim3 and accepts the loop when enough points match.
    bool compute_sim3();
    bool search_loop_map_points();
    void release_loop_candidates(bool loop_accepted);

    // Loop correction: pauses Local Mapping, propagates Scw to the current keyframe's
    // covisibles and their points, fuses the loop-side points into them, optimizes the
    // essential graph over the new links and launches the global BA thread.
    void correct_loop();
    std::map<KeyframeId, LoopConnections> loop_connections_after_fusion(const std::vector<Keyframe>& connected_keyframes);
    void search_and_fuse(const KeyframePoses& corrected_poses);

    // Global bundle adjustment (gba_thread_): the BA itself, then apply_gba_correction
    // writes the result and propagates it to what Local Mapping created meanwhile.
    void run_global_bundle_adjustment(KeyframeId loop_keyframe_id);
    void apply_gba_correction(KeyframeId loop_keyframe_id);

    // Called by run(): perform a pending reset
    void reset_if_requested();
    bool is_reset_requested() const;
    mutable std::mutex reset_mutex_;   // guards reset_requested_
    bool reset_requested_{false};

    // Called by run(): exit when asked / publish the exit
    bool is_finish_requested() const;
    void set_finished();
    mutable std::mutex finish_mutex_;   // guards finish_requested_ and finished_
    bool finish_requested_{false};
    bool finished_{true};

    FeatureType verification_feature_;        // local feature that verifies loop candidates (PlaceRecognition::verification_feature)
    std::vector<FeatureType> feature_types_;
    bool fix_scale_;                          // stereo/RGB-D: SE3 loop correction (scale fixed to 1); monocular: Sim3

    std::shared_ptr<Map> map_;
    std::shared_ptr<PlaceRecognition> place_recognition_;
    std::shared_ptr<LocalMapping> local_mapper_;
    std::shared_ptr<MapDrawer> map_drawer_;
    std::shared_ptr<FeatureMatcher> matcher_;

    mutable std::mutex new_keyframes_mutex_;   // guards new_keyframes_
    std::list<Keyframe> new_keyframes_;

    // Loop detector state (detect_loop): the keyframe being processed, the covisibility
    // groups retrieved for the previous keyframe with their consistency counts, and the
    // candidates that passed the consistency vote (input of compute_sim3)
    Keyframe current_keyframe_;
    std::vector<ConsistentGroup> consistent_groups_;
    std::vector<Keyframe> loop_candidates_;
    KeyframeId last_loop_keyframe_id_{0};   // keyId of the last closed loop's keyframe

    // Verified loop (compute_sim3, search_loop_map_points; input of correct_loop)
    Keyframe matched_keyframe_;            // the accepted loop candidate
    g2o::Sim3 g2o_Scw_;                    // world -> current keyframe through the loop side (Scm * Smw)
    mat4f Scw_;                            // the same as a 4x4 [sR | t]
    std::vector<Pt> loop_matched_points_;  // per keypoint of the current keyframe (verification_feature_): the loop-side point matched to it, or null
    std::vector<Pt> loop_map_points_;      // map points of matched_keyframe_ and its covisibles

    // Global BA thread (correct_loop launches it, run_global_bundle_adjustment runs in it)
    mutable std::mutex gba_mutex_;   // guards gba_running_, gba_stop_, gba_generation_; held during apply_gba_correction
    bool gba_running_{false};
    bool gba_stop_{false};           // read by g2o (setForceStopFlag): aborts a BA superseded by a newer loop closure
    int gba_generation_{0};          // bumped when a running BA is superseded; the BA drops its result if it changed (issue #35)
    std::thread gba_thread_;
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_LOOP_CLOSING_H
