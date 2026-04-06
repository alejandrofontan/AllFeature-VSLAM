#ifndef AF_VSLAM_FEATURE_KAZE64_H
#define AF_VSLAM_FEATURE_KAZE64_H

#include "Feature.h"
#include "FeatureExtractor.h"

namespace AF_VSLAM {

    class FeatureExtractor_kaze64;

    class Kaze64 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_KAZE64; }
        const MatcherType getMatcherType()     const override { return BF_L2; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_L2); };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;
    private:
        inline static const std::string s_featureName    = "kaze64";
        inline static const std::string s_settingsYamlFile = "settings/kaze64_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {0, 255, 255};
    };

    class FeatureExtractor_kaze64 : public FeatureExtractor {
    public:

        cv::Ptr<cv::KAZE> kaze;

        FeatureExtractor_kaze64(std::shared_ptr<FeatureExtractorSettings> &settings_);

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;
    };

    inline std::shared_ptr<FeatureExtractor> Kaze64::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_kaze64>(settings);
    }

    float DescriptorDistance_kaze64(const cv::Mat &a, const cv::Mat &b);
}

#endif //AF_VSLAM_FEATURE_KAZE64_H
