#pragma once

#include "Feature.h"
#include "FeatureExtractor.h"
#include "brisk/brisk.h"

namespace AF_VSLAM
{

class Brisk48Extractor;

class Brisk48 : public Feature
{
public:
    const std::string& get_feature_name() const override { return s_feature_name; }
    FeatureType get_type() const override { return FEAT_BRISK48; }
    MatcherType get_matcher_type() const override { return BF_HAMMING; }
    const Eigen::Matrix<float,3,1>& get_color() const override { return s_color; }
    const std::string& get_settings_yaml_file() const override { return s_settings_yaml_file; }
    float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_HAMMING); }
    std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const override;
private:
    inline static const std::string s_feature_name = "brisk48";
    inline static const std::string s_settings_yaml_file = "settings/brisk48_settings.yaml";
    inline static const Eigen::Matrix<float,3,1> s_color = {0, 122, 122};
};

class Brisk48Extractor : public FeatureExtractor
{
public:

    cv::Ptr<cv::FeatureDetector> brisk_detector;
    cv::Ptr<cv::DescriptorExtractor> brisk_extractor;

    Brisk48Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;
};

inline std::shared_ptr<FeatureExtractor> Brisk48::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Brisk48Extractor>(settings);
}


}

