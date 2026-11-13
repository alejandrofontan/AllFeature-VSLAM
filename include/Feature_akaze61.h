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

    explicit Akaze61Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;

private:
    /// The evolution for a width x height frame, rebuilt only when that size changes. The AKAZE
    /// constructor allocates the whole scale-space pyramid from the dimensions in akaze_options_,
    /// while Create_Nonlinear_Scale_Space refills every buffer per frame without re-checking the
    /// size it was allocated for -- so the object is reusable across frames, but only of one size.
    libAKAZE::AKAZE& evolution_for(int width, int height);

    std::shared_ptr<libAKAZE::AKAZE> evolution_{};
    AKAZEOptions akaze_options_{};
    int loaded_width_{0}, loaded_height_{0};   // the size evolution_ was allocated for
};

inline std::shared_ptr<FeatureExtractor> Akaze61::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Akaze61Extractor>(settings);
}

}

