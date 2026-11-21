/**
 * Module: AllFeature-VSLAM - Feature_raco128.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.2)
 * - Version: 1.0
 * - Created: 2026-11-21
 * - License: GPLv3 License
 *
 * raco128: RaCo keypoints (Shenoi et al., 3DV 2026) with ALIKED-n16 128-d descriptors on ONNX Runtime
 * (onnx/RacoOnnx.h), matched by the LightGlue trained for that pair (LIGHTGLUE_RACO ->
 * onnx::shared_lightglue("raco") in FeatureMatcher). The model file is
 * `<features_models_dir>/raco_aliked_n16_<precision>.onnx` (`pixi run setup`).
 *
 * What RaCo adds to a keypoint, and what the host does with it:
 *   ranker score  The graph returns its 2048 best keypoints by detection score; the host keeps its budget
 *                 (max_num_features) by RANKER score, after the mask - the selection the ranker is trained for. There
 *                 is no octree distribution here.
 *   covariance    A 2x2 position covariance in pixels^2. With `features_raco_covariance` (default true) it becomes the
 *                 keypoint's sigma2 and information, the quantities that weigh its residuals in pose optimisation and
 *                 bundle adjustment and size its search gates; with false the keypoint gets the fixed sigma of 1 px
 *                 the other learned features have.
 *
 * The session is created and warmed up in prepare(), at the frame size, before the first frame. There is no
 * fixed-size (Core ML) export of this graph.
 */

#pragma once

#include <memory>
#include <vector>

#include "Feature.h"
#include "FeatureExtractor.h"
#include "onnx/RacoOnnx.h"

namespace AF_VSLAM
{

class Raco128Extractor;

class Raco128 : public Feature
{
public:
    const std::string& get_feature_name() const override { return s_feature_name; }
    FeatureType get_type() const override { return FEAT_RACO128; }
    MatcherType get_matcher_type() const override { return LIGHTGLUE_RACO; }
    const Eigen::Matrix<float,3,1>& get_color() const override { return s_color; }
    int get_max_features() const override { return s_max_features; }
    float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2); }
    std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const override;

    /// The model's file stem inside features_models_dir: 'raco_aliked_n16_fp32.onnx' or 'raco_aliked_n16_fp16.onnx',
    /// by features_precision (onnx::feature_model_path)
    inline static const std::string s_model_stem = "raco_aliked_n16";

private:
    inline static const std::string s_feature_name = "raco128";
    inline static constexpr int s_max_features{2048};   // keypoints per frame at Tracking.FeaturesFraction 1
    inline static const Eigen::Matrix<float,3,1> s_color = {255, 170, 0};
};

class Raco128Extractor : public FeatureExtractor
{
public:
    explicit Raco128Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);
    ~Raco128Extractor() override = default;

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    // Creates the session for width x height frames and warms it up at that size
    void prepare(int width, int height) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;

protected:
    // RaCo's covariance per keypoint (features_raco_covariance), else the base's fixed sigma
    void compute_sigma(KeypointData& data) override;

private:
    /// Create the session on `path` and warm it up on a blank width x height frame.
    void load(const std::string& path, int width, int height);

    std::unique_ptr<onnx::RacoOnnx> extractor_;
    // Of the keypoints the last detect_and_compute returned, in their order: read by compute_sigma in the same
    // FeatureExtractor::operator() call
    std::vector<cv::Vec3f> covariances_;
};

inline std::shared_ptr<FeatureExtractor> Raco128::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Raco128Extractor>(settings);
}


}
