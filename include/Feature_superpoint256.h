/**
 * Module: AllFeature-VSLAM - Feature_superpoint256.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 2.0 (ONNX Runtime; the TensorRT engine was removed)
 * - Created: 2024-06-07
 * - Updated: 2026-10-16
 * - License: GPLv3 License
 *
 * superpoint256: SuperPoint keypoints + 256-d descriptors on ONNX Runtime (onnx/SuperPointOnnx.h), matched by
 * LightGlue-SuperPoint (LIGHTGLUE_SUPERPOINT -> onnx::shared_lightglue("superpoint") in FeatureMatcher). The model
 * file is `<features_models_dir>/superpoint_v1_<precision>.onnx` (exported locally by `pixi run setup`: the Magic
 * Leap weights are not redistributable); the runtime provider comes from the features_* settings. The session is
 * created and warmed up in prepare(), at the frame size, before the first frame. On the Core ML provider with
 * `features_static_shapes` the fixed-size export `superpoint_v1_<W>x<H>_<precision>.onnx` for that size is loaded
 * instead (Core ML compiles only bounded shapes; `export-features --only superpoint --static WxH`); the other
 * providers always load the dynamic file.
 */

#pragma once

#include <memory>

#include "Feature.h"
#include "FeatureExtractor.h"
#include "onnx/SuperPointOnnx.h"

namespace AF_VSLAM
{

class Superpoint256Extractor;

class Superpoint256 : public Feature
{
public:
    const std::string& get_feature_name() const override { return s_feature_name; }
    FeatureType get_type() const override { return FEAT_SUPERPOINT256; }
    MatcherType get_matcher_type() const override { return LIGHTGLUE_SUPERPOINT; }
    const Eigen::Matrix<float,3,1>& get_color() const override { return s_color; }
    int get_max_features() const override { return s_max_features; }
    float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override{ return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2); }
    std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const override;

    /// The model's file stem inside features_models_dir: 'superpoint_v1_fp32.onnx' or 'superpoint_v1_fp16.onnx', by
    /// features_precision (onnx::feature_model_path)
    inline static const std::string s_model_stem = "superpoint_v1";

private:
    inline static const std::string s_feature_name = "superpoint256";
    inline static constexpr int s_max_features{2048};   // keypoints per frame at Tracking.FeaturesFraction 1
    inline static const Eigen::Matrix<float,3,1> s_color = {122, 0, 255};
};

class Superpoint256Extractor : public FeatureExtractor
{
public:
    explicit Superpoint256Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);
    ~Superpoint256Extractor() override = default;

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    // Creates the session for width x height frames and warms it up at that size
    void prepare(int width, int height) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;

private:
    /// Create the session on `path` and warm it up on a blank width x height frame.
    void load(const std::string& path, int width, int height);
    /// The session for a width x height frame: the one loaded, or a new one when none is loaded yet or the loaded
    /// file is a fixed-size export for another size.
    onnx::SuperPointOnnx& extractor_for(int width, int height);

    std::unique_ptr<onnx::SuperPointOnnx> extractor_;
    bool fixed_size_{false};                    // the loaded file is a fixed-size export (Core ML with static shapes)
    int loaded_width_{0}, loaded_height_{0};   // the size the session was created and warmed up for
};

inline std::shared_ptr<FeatureExtractor> Superpoint256::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Superpoint256Extractor>(settings);
}


}

