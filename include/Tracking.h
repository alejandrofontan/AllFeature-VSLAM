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


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <functional>
#include <optional>
#include <mutex>
#include <deque>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "FeatureVocabulary.h"
#include "KeyFrameDatabase.h"
#include "FeatureExtractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include"FeatureMatcher.h"
#include "Utils.h"
#include "TrackingLostException.h"

namespace AF_VSLAM
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

// Tunable tracking heuristics, loaded from the settings YAML at System startup.
// Same pattern as OptimizerParameters: compiled-in defaults, overridden only by
// keys present in the file, so settings YAMLs without a Tracking.* block keep
// working unchanged.
struct TrackingParameters
{
    // monocular_initialization()
    int init_min_keypoints{100};     // min keypoints to accept a frame for two-view init
    float init_sigma{1.0f};          // measurement sigma for the two-view initializer
    int init_min_matches{100};       // min correspondences to attempt initialization
    int init_ransac_iterations{200}; // initializer RANSAC iteration budget
    float init_min_median_disparity{10.0f}; // px; below this (static camera) skip the init attempt

    // create_initial_map_monocular()
    int init_gba_iterations{20};      // global-BA iteration budget on the initial map
    int init_min_tracked_points{100}; // min tracked map points in the current keyframe to accept the map
    int init_min_depth_samples{10};   // min sensor-depth ratio samples to prefer depth-verified scale

    // grab_image()
    int init_extractor_features_scale{4}; // the init extractor keeps this x more keypoints

    // track_reference_keyframe()
    int track_ref_min_matches{15}; // raw matches to the reference keyframe below this -> lost
    int track_ref_min_inliers{10}; // pose-optimization inliers below this -> lost

    // track_local_map()
    int track_local_map_min_inliers{30};             // inliers against the local map below this -> lost
    int track_local_map_min_inliers_after_reloc{50}; // stricter bar shortly after a relocalization

    // update_local_keyframes()
    int max_local_keyframes{80};      // stop expanding the local map beyond this many keyframes
    int best_covisible_keyframes{10}; // covisible neighbors considered per keyframe during expansion

    // search_local_points()
    float viewing_cos_limit{0.5f}; // frustum viewing-angle limit for local-point candidates

    // need_new_keyframe()
    float min_median_flow{1.0f};        // px; below this the camera counts as static (no keyframes)
    int min_shared_points_for_flow{20}; // the flow gate needs at least this many shared points
    float ref_matches_ratio{0.9f};      // weak tracking: inliers < ratio x reference-KF tracked points...
    int min_inliers_for_keyframe{15};   // ...but above this floor (below it suggests loss, not a keyframe)
    int min_observations_high{3};       // reference-KF match counting: min observations per point
    int min_observations_low{2};        //   relaxed bar while the map is young
    int young_map_keyframes{2};         //   map considered young at <= this many keyframes
    float min_ref_overlap{0.7f};        // insert when frame/reference-keyframe overlap drops below this
    float emergency_inlier_drop_ratio{0.5f}; // emergency keyframe: inliers < ratio x recent median
    int inliers_history_size{30};       // recent-inlier reference window (frames)
    int emergency_keyframe_cooldown{10}; // min frames between emergency keyframes

    // relocalize()
    int reloc_min_matches{15};    // per-candidate matching gate
    int reloc_inliers_high{50};   // inliers to accept a pose hypothesis
    int reloc_inliers_medium{30}; // enter the narrow-window escalation
    int reloc_inliers_low{10};    // below this, abandon the hypothesis
    float reloc_search_radius_coarse{10.0f}; // first projection-search window
    float reloc_search_radius_narrow{3.0f};  // second, near-converged window
    float reloc_ransac_probability{0.99f};   // P4P RANSAC (PnPsolver)
    int reloc_ransac_min_inliers{10};
    int reloc_ransac_max_iterations{3000};
    float reloc_ransac_epsilon{0.5f};
};

class Tracking
{

public:
    Tracking(std::shared_ptr<Vocabulary> vocabulary, std::shared_ptr<FrameDrawer> frame_drawer,
             std::shared_ptr<MapDrawer> map_drawer, std::shared_ptr<Map> map,
             std::shared_ptr<KeyFrameDatabase> keyframe_db,
             const std::string& calibration_yaml, const std::string& settings_yaml,
             const std::map<FeatureType, std::string>& feature_settings_yaml_file,
             const std::vector<FeatureType>& feature_types,
             const bool fix_image_size = false);

    // Preprocess the input, extract features, and run track() on the frame.
    mat4f grab_image(Image &im, double timestamp);

    void set_local_mapper(std::shared_ptr<LocalMapping> local_mapper) { local_mapper_ = std::move(local_mapper); }
    void set_loop_closing(std::shared_ptr<LoopClosing> loop_closing) { loop_closing_ = std::move(loop_closing); }
    void set_viewer(std::shared_ptr<Viewer> viewer) { viewer_ = std::move(viewer); }

public:
    VerbosityLevel verbosity{LOW};

    // Tunable parameters, loaded from the settings YAML at System startup
    static TrackingParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState state_{NO_IMAGES_YET};
    eTrackingState last_processed_state_;

    // Current Frame
    Frame current_frame_;
    cv::Mat gray_image_;
    cv::Mat mask_image_; // segmentation mask of the current image (1 = static, 0 = dynamic); empty when segmentation is off
    std::string image_name_;

    // Initialization Variables (mono)
    std::vector<int> init_matches_;
    std::vector<vec3f> init_points3d_;
    Frame initial_frame_;

    std::map<FeatureType, std::vector<int>> matches_per_feature_;

    void reset();

    size_t num_tracked_frames_{2};

    vector<FeatureType> feature_types_{};

    int get_image_width() const {return image_width_;};
    int get_image_height() const {return image_height_;};

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // Profiling
    std::map<int, int> resize_times_{};
    std::map<int, int> frame_times_{};
    std::map<int, int> tracking_times_{};
    std::map<int, int> track_ref_times_{};
    std::map<int, int> pose_opt_times_{};
    std::map<int, int> local_map_times_{};
    std::map<int, int> grab_image_times_{};

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////

protected:

    // Main tracking function. It is independent of the input sensor.
    void track();

    // Runs one tracking stage: a TrackingLostException fails the stage (logged
    // with its reason and statistics).
    bool run_tracking_stage(const std::function<bool()>& stage);

    // Low-rate tracking-stats heartbeat for post-mortems (PROFILING_EXHAUSTIVE
    // builds only; compiles to a no-op otherwise).
    void log_heartbeat();

    // Cumulative stage-timing histograms via the AF_PROFILE sink (also
    // PROFILING_EXHAUSTIVE-only).
    void log_profile();

    // CSV of per-match residuals at the seed pose when pose optimization
    // collapses (track_reference_keyframe's divergence rescue); diagnostics only.
    void dump_pose_collapse(const mat4f& seed_pose);

    // Store this frame's pose relative to its reference keyframe (Tlr); composed
    // with the keyframe's CURRENT pose it re-derives the frame's pose next frame
    // for the pose-optimization seed, absorbing BA/loop-closure corrections.
    // Lost frames store nothing — the seed is only consumed when the previous
    // frame tracked OK.
    void store_relative_pose();

    // Map initialization for monocular: attempt_* holds the gate logic (early
    // returns); the wrapper always finishes the frame (drawer update, and the
    // trajectory entry once initialized).
    void monocular_initialization();
    void attempt_monocular_initialization();
    void create_initial_map_monocular();

    void check_replaced_in_last_frame();
    bool track_reference_keyframe();

    bool relocalize();
    // One RANSAC pose hypothesis for a relocalization candidate: seed the frame
    // with the hypothesis' inlier matches, optimize, escalate through coarse and
    // narrow projection searches; true when params.reloc_inliers_high inliers support it.
    // (candidate by value: the legacy matcher APIs take non-const Keyframe&)
    bool accept_relocalization_hypothesis(Keyframe candidate, const std::vector<Pt>& matches,
                                          const std::vector<bool>& inliers, FeatureType feature_type);

    void update_local_map();
    void update_local_points();
    void update_local_keyframes();

    bool track_local_map();
    void search_local_points();

    bool need_new_keyframe();
    void create_new_keyframe();

    // Median pixel displacement of map points shared between current_frame_ and last_frame_.
    // Scale-free stationarity signal for need_new_keyframe; empty if too few shared
    // points to be meaningful (the gate then stays inactive).
    std::optional<float> median_flow_from_last_frame() const;

    // Load intrinsics/distortion/fps for the settings' cam_mono camera from the
    // calibration yaml (hard error if the camera is missing); applies the
    // nominal-area rescale when fix_image_size_ is set.
    void load_camera_parameters(const std::string& calibration_yaml, const std::string& settings_yaml);
    // Build a feature extractor from the per-feature settings yaml, with its
    // maxNumFeatures scaled by scale_num_features (initialization uses a denser set).
    static std::shared_ptr<FeatureExtractor> get_feature_extractor(int scale_num_features,
                                                                   const std::string& feature_settings_yaml,
                                                                   FeatureType feature_type);

    //Other Thread Pointers
    std::shared_ptr<LocalMapping> local_mapper_;
    std::shared_ptr<LoopClosing> loop_closing_;

    // Features
    std::map<FeatureType, shared_ptr<FeatureExtractor>> feature_extractor_left_;
    std::map<FeatureType, shared_ptr<FeatureExtractor>> init_feature_extractor_;

    //BoW
    std::shared_ptr<Vocabulary> vocabulary_;
    shared_ptr<KeyFrameDatabase> keyframe_db_;

    // Initalization (only for monocular)
    shared_ptr<Initializer> initializer_;

    //Local Map
    Keyframe ref_keyframe_;
    std::vector<Keyframe> local_keyframes_;
    std::vector<Pt> local_points_;

    //Drawers
    std::shared_ptr<Viewer> viewer_;
    std::shared_ptr<FrameDrawer> frame_drawer_;
    std::shared_ptr<MapDrawer> map_drawer_;

    // Map
    shared_ptr<Map> map_;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    // Stereo baseline x fx. Nothing sets it yet (its only writer was the dead
    // ChangeCalibration); zero-initialized so Frame gets a defined value.
    float mbf{0.0f};
    int image_width_{};
    int image_height_{};

    // New KeyFrame rules (according to fps)
    size_t max_frames_;
    float fps_{0.0f};

    // Close/far point threshold (depth units). Nothing sets it yet; zero-initialized
    // so Frame gets a defined value.
    float mThDepth{0.0f};

    //Current matches in frame
    int num_inlier_matches_{0};

    // Rolling inlier history (last params.inliers_history_size tracked frames) — reference for the
    // emergency-keyframe trigger in need_new_keyframe(). Comparing against recent frames
    // instead of ref_keyframe_->tracked_map_points() avoids the self-inflating feedback loop
    // where every inserted keyframe grows the reference stat via post-hoc triangulation
    // (see CLAUDE.md, Stop-Induced Keyframe Runaway Investigation).
    std::deque<int> recent_inliers_history_;
    FrameId last_emergency_keyframe_id_{0};

    // Last Frame, KeyFrame and Relocalisation Info
    Keyframe last_keyframe_;
    Frame last_frame_;
    mat4f last_frame_relative_pose_; // Tlr of last_frame_ (see store_relative_pose)
    KeyframeId last_keyframe_id_;
    FrameId last_reloc_frame_id_{0};

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool is_rgb_{true};

    // Fix image size to nominal size 307200 pixels
    bool fix_image_size_{false};

    //////////////////////////////////////////////// Constants

    //////////////////////////////////////////////// Compile-time constants
    // (tunable heuristics live in TrackingParameters above)

    // update_local_keyframes()
    static constexpr size_t LOCAL_KEYFRAMES_RESERVE_SCALE{3};

    // relocalize(): P4P protocol size, chi2(0.95, 2 dof), RANSAC round scheduling
    static constexpr int RANSAC_MIN_SET{4};
    static constexpr float RANSAC_TH2{5.991f};
    static constexpr int RANSAC_ITERATIONS_PER_ROUND{5};

    // load_camera_parameters()
    static constexpr float DEFAULT_FPS{30.0f};
    static constexpr float NOMINAL_IMAGE_AREA{307200.0f}; // ~640x480, fix_image_size_ target

    std::shared_ptr<FeatureMatcher> matcher_;

    bool emergency_keyframe_{false};

};

} //namespace ORB_SLAM

#endif // TRACKING_H
