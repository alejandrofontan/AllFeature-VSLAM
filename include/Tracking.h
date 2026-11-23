/**
 * Module: AllFeature-VSLAM - Tracking.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Opus 5.5)
 * - Version: 1.1
 * - Created: 2026-01-22
 * - Updated: 2026-11-24 (Tracking review: run state, report scopes, include diet)
 * - License: GPLv3 License
 */
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

#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "Calibration.h"
#include "Frame.h"
#include "KeyframeInformation.h"
#include "Types.h"

namespace AF_VSLAM
{

// Held by pointer only: their headers stay out of this one
class FeatureMatcher;
class FrameDrawer;
class Initializer;
class LocalMapping;
class LoopClosing;
class Map;
class MapDrawer;
class Viewer;

// Tracking.FrameTracking: the frame-to-frame stage
enum class FrameTracking { reference_keyframe, optical_flow };

// Tracking.* settings: compiled-in defaults, overridden by the keys present in the settings yaml
struct TrackingParameters
{
    // Run
    bool sequential{false};   // deterministic: tracking waits for Local Mapping after every frame (issue #16)

    // Frames (grab_image)
    // keypoints per frame = this share (0..1) of each feature's get_max_features(); the viewer's slider edits it
    std::atomic<float> features_fraction{0.5f};
    float nominal_image_area{307200.0f};  // px (640x480): frames resized to this area (aspect kept), cropped to 32
    int image_width{0};                   // both > 0: resize to exactly this size instead (multiples of 32)
    int image_height{0};

    // Initialization (attempt_monocular_initialization, create_initial_map)
    int init_min_keypoints{100};             // keypoints a frame needs to take part
    float init_sigma{1.0f};                  // two-view initializer measurement sigma
    int init_min_matches{100};               // matches (all features) to attempt
    int init_ransac_iterations{200};
    float init_min_median_disparity{10.0f};  // px: below this (static camera) no attempt
    int init_gba_iterations{20};             // global BA on the initial map
    int init_min_tracked_points{100};        // fewer in the current keyframe: degenerate, reset
    int init_min_depth_samples{10};          // sensor-depth samples to take the metric scale

    // Frame to frame (track_reference_keyframe, track_with_optical_flow, optimize_pose_from_last_frame)
    FrameTracking frame_tracking{FrameTracking::reference_keyframe};   // optical_flow falls back to the reference match
    float flow_association_radius{2.0f};  // px: a flowed point takes the nearest keypoint within this
    float flow_max_backward_error{1.0f};  // px: forward-backward KLT disagreement above this drops the track
    int flow_min_matches{30};             // fewer flow matches: fall back to the reference-keyframe match
    int flow_window{21};                  // px: KLT window side
    int flow_pyramid_levels{3};           // KLT pyramid levels (with the window: the largest motion followed)
    int track_ref_min_matches{15};        // fewer matches to the reference keyframe: lost
    int track_ref_min_inliers{10};        // fewer pose inliers: lost (flow: fall back)
    int pose_rescue_min_matches{45};      // a pose collapse with this many matches retries without depth (RGB-D)

    // Local map (track_local_map)
    int track_local_map_min_inliers{30};              // fewer inliers: lost
    int track_local_map_min_inliers_after_reloc{50};  // the same within the relocalization window
    int max_local_keyframes{80};                      // the local map stops growing here
    int best_covisible_keyframes{10};                 // covisible neighbours considered per keyframe
    float viewing_cos_limit{0.5f};                    // a candidate point's viewing-angle cosine limit
    float projection_gate_sigma2{64.0f};              // squared sigmas: the projection search gate (8 sigma)

    // Keyframes (need_new_keyframe)
    float keyframe_min_information{0.05f};   // unexplained information below this: redundant
    bool log_keyframe_information{false};    // one decision line per tracked frame
    float emergency_inlier_drop_ratio{0.5f}; // busy Local Mapping: keyframe anyway when inliers < ratio x recent median
    int inliers_history_size{30};            // frames in that median
    int emergency_keyframe_cooldown{10};     // frames between emergency keyframes
    int reloc_embargo_bypass_inliers{100};   // no keyframe within the relocalization window unless this many inliers
    int reloc_embargo_min_keyframes{30};     //   and only above this many keyframes in the map

    // Loss and relocalization (track, relocalize)
    float reloc_window_seconds{1.0f};       // after a relocalization: the stricter local-map bar and the embargo
    int reset_lost_max_keyframes{0};        // lost with at most this many keyframes: reset (0 = never; no VPR: always)
    int reloc_min_matches{15};              // matches a candidate needs for a PnP solver
    int reloc_inliers_high{50};             // inliers that accept a pose
    int reloc_inliers_medium{30};           // above this after the first search: search once more
    int reloc_inliers_low{10};              // below this: give the hypothesis up
    float reloc_ransac_probability{0.99f};  // P4P RANSAC (PnPsolver)
    int reloc_ransac_min_inliers{10};
    int reloc_ransac_max_iterations{3000};
    float reloc_ransac_epsilon{0.5f};
};

class Tracking
{
public:
    // ========================================================================
    // Interface
    // ========================================================================

    Tracking(std::shared_ptr<PlaceRecognition> place_recognition, std::shared_ptr<FrameDrawer> frame_drawer,
             std::shared_ptr<MapDrawer> map_drawer, std::shared_ptr<Map> map,
             const std::string& calibration_yaml, const std::string& cam_name,
             const std::vector<FeatureType>& feature_types,
             bool fix_image_size = false);

    // Preprocess the image, extract features, track the frame; returns its pose (Tcw)
    mat4f grab_image(Image &im, double timestamp);

    void set_local_mapper(std::shared_ptr<LocalMapping> local_mapper) { local_mapper_ = std::move(local_mapper); }
    void set_loop_closing(std::shared_ptr<LoopClosing> loop_closing) { loop_closing_ = std::move(loop_closing); }
    void set_viewer(std::shared_ptr<Viewer> viewer) { viewer_ = std::move(viewer); }
    // The sole keyframe-insertion authority (KeyframeInformation.h)
    void set_keyframe_information(std::shared_ptr<KeyframeInformation> keyframe_information) { keyframe_information_ = std::move(keyframe_information); }

    void reset();
    // Drops every reference into the map (System::Shutdown, after tracking): the tracker must not track again
    void release_map_references();
    // An OK tracker becomes LOST and relocalizes on the next frame (System::Track after a pause); tracking thread only
    void request_relocalization();

    static TrackingParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

    // ========================================================================
    // Read by System, FrameDrawer and Viewer
    // ========================================================================

    std::atomic<TrackingState> state_{TrackingState::NO_IMAGES_YET};   // the Viewer thread reads it
    TrackingState last_processed_state_{TrackingState::NO_IMAGES_YET}; // before this frame (FrameDrawer)

    Frame current_frame_;
    Frame initial_frame_;          // two-view reference, drawn while NOT_INITIALIZED
    std::map<FeatureType, std::vector<int>> matches_per_feature_;   // to initial_frame_, drawn while NOT_INITIALIZED
    cv::Mat color_image_;          // the frame as loaded (BGR/BGRA or grey), after resize and crop
    cv::Mat mask_image_;           // segmentation mask (1 = static, 0 = dynamic); empty without segmentation

    static constexpr size_t FOUNDING_FRAMES{2};                // the initial map's two frames count as tracked
    std::atomic<size_t> num_tracked_frames_{FOUNDING_FRAMES};  // the Viewer thread reads it
    std::map<int, int> grab_image_times_{};                    // ms histogram of tracked frames (the viewer's median)

protected:
    // ========================================================================
    // Per-frame stages, in track() order
    // ========================================================================

    // The frame's pose (Tcw); current_frame_ is moved into last_frame_ at the end
    mat4f track();
    // A stage's exception (TrackingLostException or any other) fails it: the frame goes LOST
    bool run_tracking_stage(const std::function<bool()>& stage);
    // Tlr of an OK frame: next frame's seed is it on the reference keyframe's current pose
    void store_relative_pose();

    // Initialization: attempt_* holds the gates; the wrapper finishes the frame
    void monocular_initialization();
    void attempt_monocular_initialization();
    void create_initial_map();
    void rescale_initial_map(const Keyframe& keyframe_ini, const Keyframe& keyframe_cur, float scale) const;

    // Frame to frame
    bool track_reference_keyframe();
    bool track_with_optical_flow();
    void check_replaced_in_last_frame();
    int optimize_pose_from_last_frame(int num_matches, const char* stage);
    void discard_pose_outliers();

    // Relocalization
    bool relocalize();
    bool accept_relocalization_hypothesis(const Keyframe& candidate, const std::vector<Pt>& matches,
                                          const std::vector<bool>& inliers, FeatureType feature_type);

    // Local map
    bool track_local_map();
    void update_local_keyframes();
    void update_local_points();
    std::vector<Pt> search_local_points();   // returns every point visible in the frame
    void report_pose_inliers() const;
    // Within reloc_window_frames_ of the last relocalization (never before the first one)
    bool shortly_after_relocalization() const {
        return run_.last_reloc_frame_id && current_frame_.frame_id < *run_.last_reloc_frame_id + reloc_window_frames_;
    }

    // Keyframes
    bool need_new_keyframe();
    void log_keyframe_decision(bool insert, const std::string& reason, bool emergency,
                               const std::optional<KeyframeInformationValue>& information, float tau,
                               int median_recent_inliers) const;
    void create_new_keyframe();
    // Until Local Mapping accepts keyframes (drain_queue: and has dequeued every one), stops or finishes; map lock released
    void wait_for_local_mapper(bool drain_queue) const;

    // Setup (Tracking_aux.cc)
    float load_camera_parameters(const std::string& calibration_yaml, const std::string& cam_name);   // returns fps
    static int features_budget(FeatureType feature_type, float features_fraction);
    void apply_features_fraction();   // a changed Features Fraction reaches every extractor

    // ========================================================================
    // Collaborators
    // ========================================================================

    std::shared_ptr<Map> map_;
    std::shared_ptr<LocalMapping> local_mapper_;
    std::shared_ptr<LoopClosing> loop_closing_;
    std::shared_ptr<PlaceRecognition> place_recognition_;          // relocalization queries
    std::shared_ptr<KeyframeInformation> keyframe_information_{};  // System refuses to start without one
    std::shared_ptr<Viewer> viewer_;
    std::shared_ptr<FrameDrawer> frame_drawer_;
    std::shared_ptr<MapDrawer> map_drawer_;
    std::shared_ptr<FeatureMatcher> matcher_;
    std::shared_ptr<Initializer> initializer_;   // set while a two-view reference exists

    // ========================================================================
    // Camera and features
    // ========================================================================

    // The calibration yaml's camera, scaled for any resize: inputs to calibration_ only
    cv::Mat K;
    cv::Mat distortion_coefficients;
    DistortionModel distortion_model_{DistortionModel::radtan};
    int image_width_{};
    int image_height_{};
    float depth_noise_coeff_{0.0028f};   // sigma_depth(z) = k*z^2 (calibration depth_noise_coeff; issue #3)
    bool fix_image_size_{false};         // resize frames (NominalImageArea or ImageWidth/Height)
    std::shared_ptr<const Calibration> calibration_;   // at the frame size; every frame is checked against it
    size_t reloc_window_frames_{0};      // params.reloc_window_seconds at the camera rate

    std::vector<FeatureType> feature_types_{};
    std::map<FeatureType, std::shared_ptr<FeatureExtractor>> feature_extractors_;
    float features_fraction_in_use_{-1.0f};

    // ========================================================================
    // Frames and local map
    // ========================================================================

    Frame last_frame_;
    cv::Mat grey_image_;        // optical_flow mode only: this frame's grey
    cv::Mat last_grey_image_;   //   and the previous frame's
    std::vector<int> init_matches_;     // initial_frame_ keypoint -> current keypoint, flat over feature types
    std::vector<vec3f> init_points3d_;  // the initializer's triangulations, same indexing
    Keyframe ref_keyframe_;
    std::vector<Keyframe> local_keyframes_;
    std::vector<Pt> local_points_;

    // ========================================================================
    // Run state
    // ========================================================================

    // reset() returns it to these defaults (run_ = {})
    struct RunState
    {
        int num_inliers{0};                                  // track_local_map's inliers of the last tracked frame
        std::deque<int> recent_inliers_history;              // the emergency reference (not the reference keyframe's
                                                             // points, which grow with every insertion)
        FrameId last_emergency_keyframe_id{0};
        bool emergency_keyframe{false};                      // inserted this frame: track() waits for Local Mapping
        std::optional<FrameId> last_reloc_frame_id{};        // unset until the first relocalization
        mat4f last_frame_relative_pose{mat4f::Identity()};   // Tlr of last_frame_ (see store_relative_pose)
    };
    RunState run_{};

    // A degenerate initialization, or a loss with no way back: grab_image resets before the next frame is built
    bool reset_requested_{false};

    // ========================================================================
    // Compile-time constants (tunables live in TrackingParameters)
    // ========================================================================

    static constexpr size_t LOCAL_KEYFRAMES_RESERVE_SCALE{3};   // update_local_keyframes
    static constexpr int RANSAC_MIN_SET{4};                     // relocalize: P4P (inlier threshold: chi2_mono)
    static constexpr int RANSAC_ITERATIONS_PER_ROUND{5};
    static constexpr float DEFAULT_FPS{30.0f};                  // load_camera_parameters: no fps in the calibration
};

}
