#pragma once

#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>

#include <opencv2/core/core.hpp>

#include "Calibration.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Frame.h"
#include "FrameDrawer.h"
#include "Initializer.h"
#include "KeyframeInformation.h"
#include "PlaceRecognition.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapDrawer.h"
#include "System.h"
#include "TrackingLostException.h"
#include "Utils.h"
#include "Viewer.h"

namespace AF_VSLAM
{

// Tracking.FrameTracking: the frame-to-frame stage (see TrackingParameters::frame_tracking)
enum class FrameTracking { reference_keyframe, optical_flow };

// Broken by the include cycles above (Tracking <-> FrameDrawer/Viewer/LocalMapping/
// LoopClosing/System include each other); these forward declarations keep this
// header valid while its own include is still being processed by the peer.
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
    // track(): sequential (deterministic) mode — after any keyframe insertion the
    // tracking thread blocks until Local Mapping has fully processed it, and the
    // keyframe decision skips the live busy-flag check (issue #16). Local Mapping
    // work is thereby serialized into the tracking timeline; combined with the
    // determinism fixes (octree sort, id-ordered sets, sorted GBA input) and an
    // orb32-only / verbose:0 configuration without place recognition, runs become repeatable.
    bool sequential{false};

    // monocular_initialization()
    int init_min_keypoints{100};     // min keypoints to accept a frame for two-view init
    float init_sigma{1.0f};          // measurement sigma for the two-view initializer
    int init_min_matches{100};       // min correspondences to attempt initialization
    int init_ransac_iterations{200}; // initializer RANSAC iteration budget
    float init_min_median_disparity{10.0f}; // px; below this (static camera) skip the init attempt

    // create_initial_map()
    int init_gba_iterations{20};      // global-BA iteration budget on the initial map
    int init_min_tracked_points{100}; // min tracked map points in the current keyframe to accept the map
    int init_min_depth_samples{10};   // min sensor-depth ratio samples to prefer depth-verified scale

    // grab_image()
    // keypoints per frame = this share (0..1) of each feature's get_max_features(). Editable online (the
    // viewer's "Features Fraction" slider); Tracking applies a change before the next frame's extraction.
    std::atomic<float> features_fraction{0.5f};
    float nominal_image_area{307200.0f};  // px (640x480): frames are resized to this area (aspect kept) and cropped
                                          // to a multiple of 32; every per-frame cost scales with it
    int image_width{0};                   // both > 0: resize to exactly this size instead (intrinsics scaled per
    int image_height{0};                  // axis); multiples of 32, or the crop trims the remainder

    // Frame-to-frame stage: "reference_keyframe" = a LightGlue (or brute-force) match against the reference
    // keyframe on every frame (track_reference_keyframe); "optical_flow" = KLT on the last frame's tracked
    // points, snapped to the current frame's detected keypoints (track_with_optical_flow), with the
    // reference-keyframe match as the fallback when the flow finds too few matches or inliers
    FrameTracking frame_tracking{FrameTracking::reference_keyframe};
    float flow_association_radius{2.0f};  // px: a flowed point takes the nearest detected keypoint within this
    float flow_max_backward_error{1.0f};  // px: forward-backward KLT disagreement above this drops the track
    int flow_min_matches{30};             // map points matched by flow below this -> fall back to the reference match
    int flow_window{21};                  // px: KLT window side
    int flow_pyramid_levels{3};           // KLT pyramid levels above the image (with the window: the largest motion followed)

    // track_reference_keyframe()
    int track_ref_min_matches{15}; // raw matches to the reference keyframe below this -> lost
    int track_ref_min_inliers{10}; // pose-optimization inliers below this -> lost
    int pose_rescue_min_matches{45}; // a pose collapse with at least this many matches retries without depth (RGB-D)

    // track_local_map()
    int track_local_map_min_inliers{30};             // inliers against the local map below this -> lost
    int track_local_map_min_inliers_after_reloc{50}; // stricter bar shortly after a relocalization

    // update_local_keyframes()
    int max_local_keyframes{80};      // stop expanding the local map beyond this many keyframes
    int best_covisible_keyframes{10}; // covisible neighbors considered per keyframe during expansion

    // search_local_points()
    float viewing_cos_limit{0.5f}; // frustum viewing-angle limit for local-point candidates
    float projection_gate_sigma2{64.0f}; // squared sigmas: a projected point matches keypoints within this gate (8 sigma)

    // need_new_keyframe()
    float keyframe_min_information{0.05f}; // unexplained information (placecell, [0,1]) below which the local map already explains the view: no keyframe
    bool log_keyframe_information{false};  // one diagnostic line per tracked frame with the information and the decision
    float emergency_inlier_drop_ratio{0.5f}; // emergency keyframe: inliers < ratio x recent median
    int inliers_history_size{30};       // recent-inlier reference window (frames)
    int emergency_keyframe_cooldown{10}; // min frames between emergency keyframes
    int reloc_embargo_bypass_inliers{100}; // no keyframe within the relocalization window unless this many inliers
    int reloc_embargo_min_keyframes{30};   //   and the embargo applies only above this many keyframes in the map

    // relocalize()
    float reloc_window_seconds{1.0f};   // after a relocalization: stricter local-map bar and the keyframe embargo
    int reset_lost_max_keyframes{0};    // lost with at most this many keyframes: reset (0 = never; no VPR: always)
    int reloc_min_matches{15};    // per-candidate matching gate
    int reloc_inliers_high{50};   // inliers to accept a pose hypothesis
    int reloc_inliers_medium{30}; // above this (and below high) after the first search: search once more
    int reloc_inliers_low{10};    // below this, abandon the hypothesis
    float reloc_ransac_probability{0.99f};   // P4P RANSAC (PnPsolver)
    int reloc_ransac_min_inliers{10};
    int reloc_ransac_max_iterations{3000};
    float reloc_ransac_epsilon{0.5f};
};

class Tracking
{

public:
    Tracking(std::shared_ptr<PlaceRecognition> place_recognition, std::shared_ptr<FrameDrawer> frame_drawer,
             std::shared_ptr<MapDrawer> map_drawer, std::shared_ptr<Map> map,
             const std::string& calibration_yaml, const std::string& cam_name,
             const std::vector<FeatureType>& feature_types,
             const bool fix_image_size = false);

    // Preprocess the input, extract features, and run track() on the frame.
    mat4f grab_image(Image &im, double timestamp);

    void set_local_mapper(std::shared_ptr<LocalMapping> local_mapper) { local_mapper_ = std::move(local_mapper); }
    void set_loop_closing(std::shared_ptr<LoopClosing> loop_closing) { loop_closing_ = std::move(loop_closing); }
    void set_viewer(std::shared_ptr<Viewer> viewer) { viewer_ = std::move(viewer); }
    // Keyframe information kernel (KeyframeInformation.h): the sole keyframe-insertion authority
    void set_keyframe_information(std::shared_ptr<KeyframeInformation> keyframe_information) { keyframe_information_ = std::move(keyframe_information); }

public:
    // Tunable parameters, loaded from the settings YAML at System startup
    static TrackingParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

    // Tracking state (TrackingState lives in Types.h)
    TrackingState state_{TrackingState::NO_IMAGES_YET};
    TrackingState last_processed_state_{TrackingState::NO_IMAGES_YET};

    // Current Frame
    Frame current_frame_;
    cv::Mat color_image_; // current image as loaded (BGR/BGRA, or gray), after resize/crop; drawn by the viewer
    cv::Mat grey_image_;      // current grey frame after resize/crop (optical_flow mode only)
    cv::Mat last_grey_image_; // the previous frame's, whatever its tracking state
    cv::Mat mask_image_; // segmentation mask of the current image (1 = static, 0 = dynamic); empty when segmentation is off

    // Two-view initialization reference frame (drawn by the FrameDrawer while NOT_INITIALIZED)
    Frame initial_frame_;

    std::map<FeatureType, std::vector<int>> matches_per_feature_;

    void reset();

    // Drops every reference the tracker holds into the map: its local keyframes and points, its reference
    // keyframe and its frames. Called by System::Shutdown once tracking is over, so that the lifetime summary
    // counts only what something else still holds. The tracker must not track again afterwards.
    void release_map_references();

    // Drop the current pose and relocalize on the next frame (System::Track after a pause): an OK tracker
    // becomes LOST; an uninitialized or already lost one is left as it is. Tracking thread only.
    void request_relocalization();

    static constexpr size_t FOUNDING_FRAMES{2};          // the initial map's two frames count as tracked
    size_t num_tracked_frames_{FOUNDING_FRAMES};          // read by the viewer

    std::vector<FeatureType> feature_types_{};


    // grab_image time histogram (ms) of tracked frames: the viewer's median
    std::map<int, int> grab_image_times_{};

protected:

    // Main tracking function. It is independent of the input sensor. Returns the frame's pose (Tcw):
    // current_frame_ is moved into last_frame_ at the end, so the caller must not read it afterwards.
    mat4f track();

    // Runs one tracking stage: a TrackingLostException fails the stage (logged
    // with its reason and statistics).
    bool run_tracking_stage(const std::function<bool()>& stage);

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
    void create_initial_map();
    void rescale_initial_map(const Keyframe& keyframe_ini, const Keyframe& keyframe_cur, float scale) const;

    void check_replaced_in_last_frame();
    bool track_reference_keyframe();
    bool track_with_optical_flow();
    int optimize_pose_from_last_frame(int num_matches, const char* stage);
    void discard_pose_outliers();

    bool relocalize();
    bool accept_relocalization_hypothesis(const Keyframe& candidate, const std::vector<Pt>& matches,
                                          const std::vector<bool>& inliers, FeatureType feature_type);

    void update_local_points();
    void update_local_keyframes();

    bool track_local_map();
    std::vector<Pt> search_local_points();   // returns every point visible in the frame
    void report_pose_inliers() const;

    bool need_new_keyframe();
    void create_new_keyframe();
    // Waits for Local Mapping to accept keyframes (drain_queue: and to have dequeued every one), or to stop or finish.
    // Call with the map mutex released.
    void wait_for_local_mapper(bool drain_queue) const;
    void log_keyframe_decision(bool insert, const std::string& reason, bool emergency,
                               const std::optional<KeyframeInformationValue>& information, float tau,
                               int median_recent_inliers) const;

    float load_camera_parameters(const std::string& calibration_yaml, const std::string& cam_name);   // returns fps
    static std::shared_ptr<FeatureExtractor> get_feature_extractor(int max_num_features, FeatureType feature_type);
    static int features_budget(FeatureType feature_type, float features_fraction);
    void apply_features_fraction();
    float features_fraction_in_use_{-1.0f};

    //Other Thread Pointers
    std::shared_ptr<LocalMapping> local_mapper_;
    std::shared_ptr<LoopClosing> loop_closing_;

    // Features
    std::map<FeatureType, std::shared_ptr<FeatureExtractor>> feature_extractors_;

    // Visual place recognition backend (relocalization queries; keyframes register
    // themselves in its database)
    std::shared_ptr<PlaceRecognition> place_recognition_;

    // Keyframe information kernel: unexplained information of the current view given
    // the local keyframes (need_new_keyframe), decision history for the plots. System
    // refuses to start without one.
    std::shared_ptr<KeyframeInformation> keyframe_information_{};

    // Initialization (only for monocular)
    std::shared_ptr<Initializer> initializer_;
    std::vector<int> init_matches_;     // initial_frame_ keypoint -> current keypoint, flat over feature types
    std::vector<vec3f> init_points3d_;  // the initializer's triangulations, same indexing

    //Local Map
    Keyframe ref_keyframe_;
    std::vector<Keyframe> local_keyframes_;
    std::vector<Pt> local_points_;

    //Drawers
    std::shared_ptr<Viewer> viewer_;
    std::shared_ptr<FrameDrawer> frame_drawer_;
    std::shared_ptr<MapDrawer> map_drawer_;

    // Map
    std::shared_ptr<Map> map_;

    std::shared_ptr<FeatureMatcher> matcher_;

    // The camera as the calibration yaml declares it, scaled for any resize. These are the
    // INPUTS to the Calibration below, which owns everything derived from them; nothing but
    // load_camera_parameters and that construction should read them.
    cv::Mat K;
    cv::Mat distortion_coefficients;
    DistortionModel distortion_model_{DistortionModel::radtan};
    // The camera and everything derived from it (Calibration.h), at the frame size (after any resize and crop).
    // One instance serves the run.
    std::shared_ptr<const Calibration> calibration_;
    int image_width_{};
    int image_height_{};

    // sigma_depth(z) = k*z^2, from calibration.yaml's optional depth_noise_coeff. The default is
    // the uncalibrated Kinect-era estimate this used to hardcode in Frame (issue #3).
    float depth_noise_coeff_{0.0028f};

    // Frames in params.reloc_window_seconds at the camera rate
    size_t reloc_window_frames_{0};

    // Per-run state: reset() returns it to these defaults (run_ = {})
    struct RunState
    {
        int num_inliers{0};                                  // track_local_map's inliers of the last tracked frame
        std::deque<int> recent_inliers_history;              // last InliersHistorySize tracked frames: the emergency
                                                             // reference (not the reference keyframe's points, which
                                                             // grow with every insertion)
        FrameId last_emergency_keyframe_id{0};
        bool emergency_keyframe{false};                      // inserted this frame: track() waits for Local Mapping
        std::optional<FrameId> last_reloc_frame_id{};        // unset until the first relocalization
        mat4f last_frame_relative_pose{mat4f::Identity()};   // Tlr of last_frame_ (see store_relative_pose)
    };
    RunState run_{};

    // A degenerate initialization, or a loss with no way back: grab_image resets before the next frame is built
    bool reset_requested_{false};

    Frame last_frame_;
    // Within reloc_window_frames_ of the last relocalization (never before the first one)
    bool shortly_after_relocalization() const {
        return run_.last_reloc_frame_id && current_frame_.frame_id < *run_.last_reloc_frame_id + reloc_window_frames_;
    }

    // Resize frames (Tracking.NominalImageArea or Tracking.ImageWidth/Height; see load_camera_parameters)
    bool fix_image_size_{false};

    //////////////////////////////////////////////// Constants

    //////////////////////////////////////////////// Compile-time constants
    // (tunable heuristics live in TrackingParameters above)

    // update_local_keyframes()
    static constexpr size_t LOCAL_KEYFRAMES_RESERVE_SCALE{3};

    // relocalize(): P4P protocol size, RANSAC round scheduling (the inlier threshold is ErrorModel::params.chi2_mono)
    static constexpr int RANSAC_MIN_SET{4};
    static constexpr int RANSAC_ITERATIONS_PER_ROUND{5};

    // load_camera_parameters()
    static constexpr float DEFAULT_FPS{30.0f};
};

}
