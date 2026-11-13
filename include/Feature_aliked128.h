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
 * `<features_models_dir>/aliked_n16.onnx` (`pixi run download-feature-models`). The graph detects in top-k mode
 * (its 2048 best keypoints whatever their score; a threshold-mode export, 'aliked_n16_threshold.onnx', exists for
 * comparison); the host's octree distribution brings them down to the keypoint budget (max_num_features), so the
 * budget is what sets the number of keypoints now. With `features_static_shapes` the fixed-size export
 * `aliked_n16_<W>x<H>.onnx` for the frame size is loaded on the first frame instead (Core ML;
 * `export-features --only aliked --static WxH`).
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
    const std::string& get_settings_yaml_file() const override { return s_settings_yaml_file; }
    float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2); }
    std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const override;

    /// The model file inside features_models_dir (fp32 export of tools/export_features.py)
    inline static const std::string s_model_file = "aliked_n16.onnx";

private:
    inline static const std::string s_feature_name = "aliked128";
    inline static const std::string s_settings_yaml_file = "settings/aliked128_settings.yaml";
    inline static const Eigen::Matrix<float,3,1> s_color = {181, 243, 249};
};

class Aliked128Extractor : public FeatureExtractor
{
public:
    explicit Aliked128Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);
    ~Aliked128Extractor() override = default;

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;

private:
    /// Create the session on `path` and warm it up on a blank width x height frame.
    void load(const std::string& path, int width, int height);
    /// The session for a width x height frame: the one loaded, or (static shapes) the export for that size.
    onnx::AlikedOnnx& extractor_for(int width, int height);

    std::unique_ptr<onnx::AlikedOnnx> extractor_;
    int loaded_width_{0}, loaded_height_{0};   // the static export's size (static shapes only)
};

inline std::shared_ptr<FeatureExtractor> Aliked128::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Aliked128Extractor>(settings);
}


}

