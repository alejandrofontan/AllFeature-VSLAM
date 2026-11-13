#pragma once

#include <opencv2/features2d.hpp>

#include "Feature.h"
#include "FeatureExtractor.h"

namespace AF_VSLAM
{

class Kaze64Extractor;

class Kaze64 : public Feature
{
public:
    const std::string& get_feature_name() const override { return s_feature_name; }
    FeatureType get_type() const override { return FEAT_KAZE64; }
    MatcherType get_matcher_type() const override { return BF_L2; }
    const Eigen::Matrix<float,3,1>& get_color() const override { return s_color; }
    const std::string& get_settings_yaml_file() const override { return s_settings_yaml_file; }
    float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_L2); }
    std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const override;
private:
    inline static const std::string s_feature_name = "kaze64";
    inline static const std::string s_settings_yaml_file = "settings/kaze64_settings.yaml";
    inline static const Eigen::Matrix<float,3,1> s_color = {0, 255, 255};
};

class Kaze64Extractor : public FeatureExtractor
{
public:

    cv::Ptr<cv::KAZE> kaze;

    Kaze64Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;
};

inline std::shared_ptr<FeatureExtractor> Kaze64::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Kaze64Extractor>(settings);
}

}

