/**
 * Module: AllFeature-VSLAM - Feature_aliked128.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 2.0 (ONNX Runtime; the libtorch port was removed)
 * - Created: 2024-06-07
 * - Updated: 2026-10-16
 * - License: GPLv3 License
 *
 * aliked128: ALIKED-n16 sub-pixel keypoints + 128-d descriptors on ONNX Runtime (onnx/AlikedOnnx.h), matched by
 * LightGlue-ALIKED (LIGHTGLUE_ALIKED -> onnx::shared_lightglue("aliked") in FeatureMatcher). The model file is
 * `<features_models_dir>/aliked_n16_<precision>.onnx` (`pixi run setup`). The graph detects in top-k mode
 * (its 2048 best keypoints whatever their score; a threshold-mode export, 'aliked_n16_threshold.onnx', exists for
 * comparison); the host's octree distribution brings them down to the keypoint budget (max_num_features), so the
 * budget is what sets the number of keypoints now. The session is created and warmed up in prepare(), at the frame
 * size, before the first frame. On the Core ML provider with `features_static_shapes` the fixed-size export
 * `aliked_n16_<W>x<H>_<precision>.onnx` for that size is loaded instead (`export-features --only aliked --static
 * WxH`); the other providers always load the dynamic file.
 */

#pragma once

#include <memory>

#include "Feature.h"
#include "FeatureExtractor.h"
#include "onnx/AlikedOnnx.h"

namespace AF_VSLAM
{

class Aliked128Extractor;

class Aliked128 : public Feature
{
public:
    const std::string& get_feature_name() const override { return s_feature_name; }
    FeatureType get_type() const override { return FEAT_ALIKED128; }
    MatcherType get_matcher_type() const override { return LIGHTGLUE_ALIKED; }
    const Eigen::Matrix<float,3,1>& get_color() const override { return s_color; }
    int get_max_features() const override { return s_max_features; }
    float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2); }
    std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const override;

    /// The model's file stem inside features_models_dir: 'aliked_n16_fp32.onnx' or 'aliked_n16_fp16.onnx', by
    /// features_precision (onnx::feature_model_path)
    inline static const std::string s_model_stem = "aliked_n16";

private:
    inline static const std::string s_feature_name = "aliked128";
    inline static constexpr int s_max_features{2048};   // keypoints per frame at Tracking.FeaturesFraction 1
    inline static const Eigen::Matrix<float,3,1> s_color = {181, 243, 249};
};

class Aliked128Extractor : public FeatureExtractor
{
public:
    explicit Aliked128Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);
    ~Aliked128Extractor() override = default;

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    // Creates the session for width x height frames and warms it up at that size
    void prepare(int width, int height) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;

private:
    /// Create the session on `path` and warm it up on a blank width x height frame.
    void load(const std::string& path, int width, int height);
    /// The session for a width x height frame: the one loaded, or a new one when none is loaded yet or the loaded
    /// file is a fixed-size export for another size.
    onnx::AlikedOnnx& extractor_for(int width, int height);

    std::unique_ptr<onnx::AlikedOnnx> extractor_;
    bool fixed_size_{false};                    // the loaded file is a fixed-size export (Core ML with static shapes)
    int loaded_width_{0}, loaded_height_{0};   // the size the session was created and warmed up for
};

inline std::shared_ptr<FeatureExtractor> Aliked128::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Aliked128Extractor>(settings);
}


}

