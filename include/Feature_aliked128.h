#ifndef AF_VSLAM_FEATURE_ALIKED128_H
#define AF_VSLAM_FEATURE_ALIKED128_H

#include "Feature.h"
#include "FeatureExtractor.h"
#include "feature/ALIKED.hpp"

namespace AF_VSLAM {

    class FeatureExtractor_aliked128;

    class Aliked128 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        FeatureType  getType()           const override { return FEAT_ALIKED128; }
        MatcherType getMatcherType()     const override { return LIGHTGLUE_ALIKED; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2); };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;
    private:
        inline static const std::string s_featureName    = "aliked128";
        inline static const std::string s_settingsYamlFile = "settings/aliked128_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {181, 243, 249};
    };

    class FeatureExtractor_aliked128 : public FeatureExtractor {
    public:

        std::shared_ptr<ALIKED> extractor;

        FeatureExtractor_aliked128(std::shared_ptr<FeatureExtractorSettings> &settings_);
        ~FeatureExtractor_aliked128() {}

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;
    };

    inline std::shared_ptr<FeatureExtractor> Aliked128::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_aliked128>(settings);
    }

    float DescriptorDistance_aliked128(const cv::Mat &a, const cv::Mat &b);

}

#endif //AF_VSLAM_FEATURE_ALIKED128_H
