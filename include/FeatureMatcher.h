#pragma once

#include <mutex>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

// Learned matching: the three LightGlue variants (superpoint / aliked / sift) run on ONNX Runtime through the
// process-wide sessions of onnx/FeatureModels.h; classical descriptors
// go through the brute-force matcher (BruteForceMatcher.h). No CMake backend option gates the dispatch any more.

namespace AF_VSLAM
{

class FeatureMatcher
{
public:

    // Loads (once per process) and warms up the LightGlue variant of every learned feature in `feature_types`.
    FeatureMatcher(const std::vector<FeatureType>& feature_types,
        std::string name = "Extractor");

    // Computes the Hamming distance between two ORB descriptors
    static Descriptor_Distance_Type descriptor_distance(const cv::Mat &a, const cv::Mat &b, const FeatureType& featureType_);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<cv::DMatch> match_descriptors(const cv::Mat& desc1, const cv::Mat& desc2,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2, const FeatureType& feat_type,
        cv::Size image_size1, cv::Size image_size2);

    static constexpr double filter_inlier_threshold_px = 3.0;  // RANSAC's Sampson error and refinement loss, pixels
    static constexpr size_t filter_min_inliers = 8;            // fewer: the model is rejected
    static constexpr size_t filter_min_iterations = 100;       // RANSAC rounds, lower and upper bound
    static constexpr size_t filter_max_iterations = 1000;
    static std::vector<cv::DMatch> filter_matches_geometric(const std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2,
        const std::vector<mat2f>& sigma2_1, const std::vector<mat2f>& sigma2_2,
        const Calibration& calibration1, const Calibration& calibration2);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::map<FeatureType, int> match_keyframe_to_frame(const Keyframe& keyframe, Frame& frame,
        std::map<FeatureType, std::vector<Pt>>& map_pts_matches,
        const std::vector<FeatureType>& feat_types);

    // Gives a map point to the frame keypoints that have none yet. `map_pts` are the candidates and
    // `projections` where each projects in the frame, parallel arrays. A candidate and a free keypoint can be
    // paired only inside the gate below; among those pairs, one is accepted when each side is the other's best
    // descriptor match. Local search, no descriptor-distance threshold. Accepted matches go to frame.pts.
    // Used in: Tracking::search_local_points, Tracking::accept_relocalization_hypothesis
    // The gate: the squared offset between keypoint and projection must be below this many times the keypoint's
    // own position variance (KeypointData::sigma2 / information), the model pose optimization tests outliers
    // with -- so a coarser keypoint gets a wider gate. 64 is (8 sigmas)^2: on ETH table_3 the correct orb32
    // matches lie within 8 px at sigma 1. It is well above the optimizer's chi2 of 5.99 because the gate also
    // has to absorb the pose's and the map point's error.
    static constexpr float projection_match_scale_sigma2 = 64.0f;
    void match_map_points_to_frame(Frame& frame, const std::vector<Pt>& map_pts,
                                   const std::vector<cv::Point2f>& projections);

    // The same search on a keyframe's keypoints of one feature type, writing into a vector the caller owns:
    // `matched_points` has one slot per keypoint of `feat_type`; a keypoint is free when its slot is empty, and an
    // accepted match fills it. The keyframe is not modified. An all-empty `matched_points` makes every keypoint a
    // target, occupied or not, which is what fusion wants. `gate_scale_sigma2` replaces the gate above, and a pair
    // whose descriptor distance exceeds `max_descriptor_distance` is dropped (only fusion sets it).
    // Used in: LoopClosing::search_loop_map_points, LoopClosing::search_and_fuse, LocalMapping::fuse_into_keyframe
    void match_map_points_to_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_pts,
                                      const std::vector<cv::Point2f>& projections,
                                      FeatureType feat_type, std::vector<Pt>& matched_points,
                                      float gate_scale_sigma2 = projection_match_scale_sigma2,
                                      Descriptor_Distance_Type max_descriptor_distance =
                                          std::numeric_limits<Descriptor_Distance_Type>::max());

    // The last frame's tracked map points carried into `frame` by optical flow, without descriptors: KLT from
    // `last_grey` to `grey` and back (round-trip error above max_backward_error px drops a track), then each
    // flowed point gives its map point to the nearest keypoint of its type within association_radius px (a
    // keypoint claimed twice keeps the closer track). Writes frame.pts. The counts: last-frame points tracked,
    // tracks that survived the round trip, keypoints matched.
    // Used in: Tracking::track_with_optical_flow
    struct FlowMatchCounts { int tracked{0}; int survived{0}; int matched{0}; };
    static constexpr int flow_window_px = 21;          // KLT window
    static constexpr int flow_pyramid_levels = 3;      // KLT pyramid levels above the image
    static constexpr int flow_max_iterations = 30;     // KLT stopping criteria
    static constexpr double flow_epsilon = 0.01;
    FlowMatchCounts match_frame_to_frame_by_flow(const Frame& last_frame, const cv::Mat& last_grey, Frame& frame,
                                                 const cv::Mat& grey, float max_backward_error, float association_radius);

    // Makes sure the matches between `keyframe` and each of `others` are in `keyframe`'s match cache: pairs
    // already cached are skipped; the rest are matched (all pairs and feature types in parallel), filtered
    // geometrically per pair, jointly over the feature types, and stored in `keyframe` only. The only writer of
    // the cache for keyframe pairs; tracking writes it for a frame against its reference keyframe, filtered too.
    // The cache is a hand-off: read the pairs back, then release them with clear_keyframe_matches.
    // Used in: LocalMapping::create_new_map_points, LoopClosing::compute_sim3, LoopClosing::search_loop_map_points.
    // The pairs are read back with get_keyframe_matches.
    void match_keyframe_to_keyframes(const Keyframe& keyframe, const std::vector<Keyframe>& others,
        const std::vector<FeatureType>& feat_types);

    // The matched keypoint pairs (index in keyframe1, index in keyframe2) that keyframe1's match cache holds for
    // keyframe2, per feature type. A read only: it never matches. A pair of keyframes that was not cached first
    // (match_keyframe_to_keyframes) gives no pairs.
    // Used in: LocalMapping::create_new_map_points, LoopClosing::compute_sim3
    static std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> get_keyframe_matches(
        const Keyframe& keyframe1, const Keyframe& keyframe2);

    // Releases every match list `keyframe` holds; the cache is otherwise never pruned. Called by the thread that
    // filled them once it has read them back.
    // Used in: LocalMapping::process_keyframe, LoopClosing::run
    static void clear_keyframe_matches(const Keyframe& keyframe);

    // The matched keypoint pairs (index in frame1, index in frame2) of two frames, per feature type, for
    // monocular initialization: descriptor matching per type, then the geometric filter on all types jointly.
    // The result has an entry for every requested feature type, empty when none of its matches survive.
    // Used in: Tracking::attempt_monocular_initialization
    std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> match_frames_for_initialization(const Frame& frame1,
        const Frame& frame2, const std::vector<FeatureType>& feat_types);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

public:

    static VerbosityLevel verbosity;

protected:

    static std::map<FeatureType, std::vector<cv::DMatch>> filter_matches_geometric_jointly(
        const std::map<FeatureType, std::vector<cv::DMatch>>& matches_by_type,
        const FeatureData& features1, const FeatureData& features2,
        const Calibration& calibration1, const Calibration& calibration2);

    string name{};

    // Matching options
    std::vector<FeatureType> feature_types;
};

}
