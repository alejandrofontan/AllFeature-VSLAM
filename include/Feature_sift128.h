/**
 * Module: AllFeature-VSLAM - Feature_sift128.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 2.0 (cv::SIFT on the CPU; the SiftGPU backend was removed)
 * - Created: 2024-06-07
 * - Updated: 2026-10-16
 * - License: GPLv3 License
 *
 * sift128: SIFT keypoints + RootSIFT descriptors on OpenCV's CPU implementation (SiftCpu.h: the LightGlue
 * repository's SIFT extractor exactly - duplicate-orientation filter, RootSIFT, scale and orientation kept), matched
 * by LightGlue-SIFT (LIGHTGLUE_SIFT -> onnx::shared_lightglue("sift") in FeatureMatcher; the descriptor-only path is
 * the cross-checked brute-force L2 like the other float descriptors). The SLAM-facing octave and size stay 0 and 1
 * as before; the scale lives in cv::KeyPoint::size and the orientation in ::angle for the matcher.
 */

#pragma once

#include <memory>

#include "Feature.h"
#include "FeatureExtractor.h"
#include "SiftCpu.h"

namespace AF_VSLAM
{

class Sift128Extractor;

class Sift128 : public Feature
{
public:
    const std::string& get_feature_name() const override { return s_feature_name; }
    FeatureType get_type() const override { return FEAT_SIFT128; }
    MatcherType get_matcher_type() const override { return LIGHTGLUE_SIFT; }
    const Eigen::Matrix<float,3,1>& get_color() const override { return s_color; }
    const std::string& get_settings_yaml_file() const override { return s_settings_yaml_file; }
    float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_L2); }
    std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const override;

private:
    inline static const std::string s_feature_name = "sift128";
    inline static const std::string s_settings_yaml_file = "settings/sift128_settings.yaml";
    inline static const Eigen::Matrix<float,3,1> s_color = {255, 0, 0};
};

class Sift128Extractor : public FeatureExtractor
{
public:
    explicit Sift128Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);
    ~Sift128Extractor() override = default;

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;

private:
    std::unique_ptr<SiftCpu> sift_;
};

inline std::shared_ptr<FeatureExtractor> Sift128::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Sift128Extractor>(settings);
}


}

