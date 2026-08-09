#ifndef AF_VSLAM_FEATURE_AKAZE61_H
#define AF_VSLAM_FEATURE_AKAZE61_H

#include "Feature.h"
#include "FeatureExtractor.h"
#include "akaze/AKAZE.h"

namespace AF_VSLAM {

    class FeatureExtractor_akaze61;

    class Akaze61 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        FeatureType  getType()           const override { return FEAT_AKAZE61; }
        MatcherType getMatcherType()     const override { return BF_HAMMING; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_HAMMING); };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;
    private:
        inline static const std::string s_featureName    = "akaze61";
        inline static const std::string s_settingsYamlFile = "settings/akaze61_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {255, 255, 0};
    };

    class FeatureExtractor_akaze61 : public FeatureExtractor {
    public:

        std::shared_ptr<libAKAZE::AKAZE> evolution{};
        AKAZEOptions akazeOptions{};

        FeatureExtractor_akaze61(std::shared_ptr<FeatureExtractorSettings> &settings_);

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        void setupImage(const Image& img) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;

    };

    inline std::shared_ptr<FeatureExtractor> Akaze61::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_akaze61>(settings);
    }

    float DescriptorDistance_akaze61(const cv::Mat &a, const cv::Mat &b);
}

#endif //AF_VSLAM_FEATURE_AKAZE61_H
