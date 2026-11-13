#pragma once

#include "Feature.h"
#include "FeatureExtractor.h"

namespace AF_VSLAM
{

class Orb32Extractor;

class Orb32 : public Feature
{
public:
    const std::string& get_feature_name() const override { return s_feature_name; }
    FeatureType get_type() const override { return FEAT_ORB32; }
    MatcherType get_matcher_type() const override { return BF_HAMMING; }
    const std::string& get_settings_yaml_file() const override { return s_settings_yaml_file; }

    const Eigen::Matrix<float,3,1>& get_color() const override { return s_color; }
    float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override;
    std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const override;
private:
    inline static const std::string s_feature_name = "orb32";
    inline static const std::string s_settings_yaml_file = "settings/orb32_settings.yaml";
    inline static const Eigen::Matrix<float,3,1> s_color = {129, 149, 251};
};

class Orb32Extractor : public FeatureExtractor
{
public:

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;
    const int num_octaves;        // pyramid levels (FeatureExtractor.num_octaves)
    const float scale_factor;   // scale between consecutive levels (FeatureExtractor.scale_factor)
    std::vector<float> scale_factors;
    std::vector<float> inv_scale_factors;
    std::vector<cv::Mat> image_pyramid;
    // Backing store for image_pyramid: each level's bordered buffer, of which image_pyramid
    // holds the inner (border-free) view. Kept between frames so a pyramid of an unchanged
    // frame size costs no allocation; see compute_pyramid.
    std::vector<cv::Mat> pyramid_buffers;
    int initial_fast_threshold{20};
    int min_fast_threshold{7};
    std::vector<int> features_per_level;
    std::vector<int> u_max;
    std::vector<cv::Point> pattern;

    Orb32Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_);

    void detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint &keypoint) const override;

    // orb32's own: the pyramid level a keypoint came from. Not on FeatureExtractor -- it was
    // a pure virtual there that only get_keypoint_size below ever called, so every other
    // extractor carried a `return 0;` override to satisfy an interface nothing used.
    [[nodiscard]] int get_keypoint_octave(const cv::KeyPoint &keypoint) const;

    void compute_pyramid(cv::Mat image);
    void compute_keypoints_oct_tree(std::vector<std::vector<cv::KeyPoint> >& all_keypoints);
};

inline std::shared_ptr<FeatureExtractor> Orb32::create_extractor(
    std::shared_ptr<FeatureExtractorSettings> settings) const {
    return std::make_shared<Orb32Extractor>(settings);
}

float DescriptorDistance_orb32(const cv::Mat &a, const cv::Mat &b);
}

