#pragma once

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

// Learned matching: the three LightGlue variants (superpoint / aliked / sift) run on ONNX Runtime through the
// process-wide sessions of onnx/FeatureModels.h; classical descriptors go through the brute-force matcher
// (BruteForceMatcher.h).

namespace AF_VSLAM
{

class FeatureMatcher
{
public:

    // Loads (once per process) and warms up the LightGlue variant of every learned feature in `feature_types`.
    FeatureMatcher(const std::vector<FeatureType>& feature_types, const std::string& name);

    // ==================== Image to Image ====================
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

    std::map<FeatureType, int> match_keyframe_to_frame(const Keyframe& keyframe, Frame& frame,
        std::map<FeatureType, std::vector<Pt>>& map_pts_matches,
        const std::vector<FeatureType>& feat_types);

    void match_keyframe_to_keyframes(const Keyframe& keyframe, const std::vector<Keyframe>& others,
        const std::vector<FeatureType>& feat_types);

    static std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> get_keyframe_matches(
        const Keyframe& keyframe1, const Keyframe& keyframe2);

    static void clear_keyframe_matches(const Keyframe& keyframe);

    std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> match_frames_for_initialization(const Frame& frame1,
        const Frame& frame2, const std::vector<FeatureType>& feat_types);

    // ==================== Map to Image ====================
    static Descriptor_Distance_Type descriptor_distance(const cv::Mat &a, const cv::Mat &b, const FeatureType& featureType_);

    static constexpr float projection_match_scale_sigma2 = 64.0f;
    void match_map_points_to_frame(Frame& frame, const std::vector<Pt>& map_pts,
                                   const std::vector<cv::Point2f>& projections);

    void match_map_points_to_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_pts,
                                      const std::vector<cv::Point2f>& projections,
                                      FeatureType feat_type, std::vector<Pt>& matched_points,
                                      float gate_scale_sigma2 = projection_match_scale_sigma2,
                                      Descriptor_Distance_Type max_descriptor_distance =
                                          std::numeric_limits<Descriptor_Distance_Type>::max());

    // ==================== Optical Flow ====================
    struct FlowMatchCounts { int tracked{0}; int survived{0}; int matched{0}; };
    static constexpr int flow_max_iterations = 30;     // KLT stopping criteria
    static constexpr double flow_epsilon = 0.01;
    FlowMatchCounts match_frame_to_frame_by_flow(const Frame& last_frame, const cv::Mat& last_grey, Frame& frame,
                                                 const cv::Mat& grey, int window, int pyramid_levels,
                                                 float max_backward_error, float association_radius);

protected:

    static std::map<FeatureType, std::vector<cv::DMatch>> filter_matches_geometric_jointly(
        const std::map<FeatureType, std::vector<cv::DMatch>>& matches_by_type,
        const FeatureData& features1, const FeatureData& features2,
        const Calibration& calibration1, const Calibration& calibration2);
};

}
