#ifndef AF_VSLAM_FEATURE_BRISK48_H
#define AF_VSLAM_FEATURE_BRISK48_H

#include "Feature.h"
#include "FeatureExtractor.h"
#include "brisk/brisk.h"

namespace AF_VSLAM {

    class FeatureExtractor_brisk48;

    class Brisk48 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_BRISK; }
        const MatcherType getMatcherType()     const override { return BF_HAMMING; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float DescriptorDistance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_HAMMING); };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;
    private:
        inline static const std::string s_featureName    = "brisk48";
        inline static const std::string s_settingsYamlFile = "settings/brisk48_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {0, 122, 122};
    };

    class FeatureExtractor_brisk48 : public FeatureExtractor {
    public:

        cv::Ptr<cv::FeatureDetector> brisk_detector;
        cv::Ptr<cv::DescriptorExtractor> brisk_extractor;

        FeatureExtractor_brisk48(std::shared_ptr<FeatureExtractorSettings> &settings_);

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;
    };

    inline std::shared_ptr<FeatureExtractor> Brisk48::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_brisk48>(settings);
    }

    float DescriptorDistance_brisk48(const cv::Mat &a, const cv::Mat &b);

}

#endif //AF_VSLAM_FEATURE_BRISK48_H
