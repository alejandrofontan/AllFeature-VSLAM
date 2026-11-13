#pragma once

#include "Feature.h"
#include "FeatureExtractor.h"
#include "akaze/AKAZE.h"

namespace AF_VSLAM
{

class Akaze61Extractor;

class Akaze61 : public Feature
{
public:
    const std::string& get_feature_name() const override { return s_feature_name; }
    FeatureType get_type() const override { return FEAT_AKAZE61; }
    MatcherType get_matcher_type() const override { return BF_HAMMING; }
    const Eigen::Matrix<float,3,1>& get_color() const override { return s_color; }
    const std::string& get_settings_yaml_file() const override { return s_settings_yaml_file; }
    float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_HAMMING); }
    std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const override;
private:
    inline static const std::string s_feature_name = "akaze61";
    inline static const std::string s_settings_yaml_file = "settings/akaze61_settings.yaml";
    inline static const Eigen::Matrix<float,3,1> s_color = {255, 255, 0};
};

class Akaze61Extractor : public FeatureExtractor
{
public:

    std::shared_ptr<libAKAZE::AKAZE> evolution{};
    AKAZEOptions akaze_options{};

    Akaze61Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    void setup_image(const Image& img) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;
};

inline std::shared_ptr<FeatureExtractor> Akaze61::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Akaze61Extractor>(settings);
}

}

