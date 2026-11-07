#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "Calibration.h"
#include "Types.h"
#include "Image.h"
#include "FeatureExtractor.h"
#include "PlaceRecognition.h"

namespace AF_VSLAM
{
using namespace std;

inline constexpr int FRAME_GRID_ROWS = 48;
inline constexpr int FRAME_GRID_COLS = 64;

class MapPoint;
typedef std::shared_ptr<AF_VSLAM::MapPoint> Pt;

class KeyFrame;
typedef std::shared_ptr<AF_VSLAM::KeyFrame> Keyframe;

class Frame
{
public:

    // ========================================================================
    // Construction
    // ========================================================================

    Frame() = default;
    Frame(const Frame&) = default;
    Frame& operator=(const Frame&) = default;
    Frame(Frame&&) = default;
    Frame& operator=(Frame&&) = default;

    Frame(const Image& img, double timestamp_,
        const std::map<FeatureType, std::shared_ptr<FeatureExtractor>>& extractors,
        std::shared_ptr<PlaceRecognition> place_recognition_,
        std::shared_ptr<const Calibration> calibration_);

    // ========================================================================
    // Pose
    // ========================================================================

    void set_pose(const mat4f& Tcw_);

    [[nodiscard]] inline const mat4f& get_pose() const {
        return Tcw;
    }

    // ========================================================================
    // Per-frame queries
    // ========================================================================

    void compute_global_descriptor();

    [[nodiscard]] int count_inlier_map_points() const;
    void drop_outlier_points();

    [[nodiscard]] std::optional<cv::Point2f> project_if_visible(const Pt& pt, float viewing_cos_limit) const;

    [[nodiscard]] float get_keypoint_sigma2(KeypointIndex keypoint_index, FeatureType feature_type) const
    {
        return 0.5f * (keyPtsSigma2.at(feature_type)[keypoint_index](0,0) + keyPtsSigma2.at(feature_type)[keypoint_index](1,1));
    }

    [[nodiscard]] mat2f get_keypoint_information_2d(KeypointIndex keypoint_index, FeatureType feature_type) const
    {
        return keyPtsInf.at(feature_type)[keypoint_index];
    }

    // ========================================================================
    // Spatial index
    // ========================================================================

    [[nodiscard]] std::vector<std::size_t> get_features_in_area(float x, float y, float r,
                                                                FeatureType feature_type) const;

    // ========================================================================
    // Data
    // ========================================================================

    std::vector<FeatureType> feature_types;
    std::map<FrameId, std::map<FeatureType, std::vector<cv::DMatch>>> cache_matched_pairs_feat_type{};

    Eigen::VectorXf global_descriptor;

    cv::Mat image;
    double timestamp{0.0};

    std::shared_ptr<const Calibration> calibration;

    std::map<FeatureType, int> N;

    std::map<FeatureType, std::vector<cv::KeyPoint>> raw_keypoints;   // distorted
    std::map<FeatureType, std::vector<cv::KeyPoint>> keypoints;       // undistorted
    std::map<FeatureType, cv::Mat> descriptors;
    std::map<FeatureType, std::vector<float>> inv_depth;
    std::map<FeatureType, std::vector<float>> sigma2_inv_depth;
    std::map<FeatureType, std::vector<cv::Vec3b>> keypoint_colors;

    std::map<FeatureType, std::vector<Pt>> pts;
    std::map<FeatureType, std::vector<bool>> outliers;

    std::map<FeatureType, std::vector<mat2f>> keyPtsSigma2{};
    std::map<FeatureType, std::vector<mat2f>> keyPtsInf{};
    std::map<FeatureType, std::vector<float>> keyPtsSize{};

    static inline FrameId next_id{0};
    FrameId frame_id{0};

    Keyframe ref_keyframe;

    float maxKeyPtSize{};

    using Grid = std::array<std::array<std::vector<std::size_t>, FRAME_GRID_ROWS>, FRAME_GRID_COLS>;

    mutable std::map<FeatureType, Grid> grid;
    void build_grids() const;

private:

    // ========================================================================
    // Construction
    // ========================================================================

    void extract_features(const Image& img,
                          const std::map<FeatureType, std::shared_ptr<FeatureExtractor>>& extractors);
    void undistort_keypoints();
    void get_depth(const Image& img);
    void get_colors(const Image& img);

    std::shared_ptr<PlaceRecognition> place_recognition;

    // ========================================================================
    // Pose
    // ========================================================================

    mat4f Tcw{mat4f::Zero()};
    mat3f Rcw{mat3f::Identity()};
    vec3f tcw{vec3f::Zero()};
    mat3f Rwc{mat3f::Identity()};
    vec3f twc{vec3f::Zero()};

    // ========================================================================
    // Spatial index
    // ========================================================================

    const Grid& grid_for(FeatureType feature_type) const;
    [[nodiscard]] cv::Point2i pos_in_grid(const cv::KeyPoint& kp) const;
};

[[nodiscard]] std::vector<std::size_t> features_in_area(const Frame::Grid& grid,
                                                        const std::vector<cv::KeyPoint>& keypoints,
                                                        const Calibration& cal,
                                                        float x, float y, float r);

}
