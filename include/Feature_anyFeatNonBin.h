#ifndef AF_VSLAM_FEATURE_ANYFEATNONBIN_H
#define AF_VSLAM_FEATURE_ANYFEATNONBIN_H

#include "Feature.h"
#include "FeatureExtractor.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace AF_VSLAM {

    class FeatureExtractor_anyFeatNonBin;

    class AnyFeatNonBin : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_ANYFEATNONBIN; }
        const MatcherType getMatcherType()     const override { return BF_L2; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return 0.0f; };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;
    private:
        inline static const std::string s_featureName    = "anyfeatnonbin";
        inline static const std::string s_settingsYamlFile = "settings/anyFeatNonBin_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {50, 100, 200};
    };

    class FeatureExtractor_anyFeatNonBin : public FeatureExtractor {
    public:

        FeatureExtractor_anyFeatNonBin(std::shared_ptr<FeatureExtractorSettings> &settings_);

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;

    };

    inline std::shared_ptr<FeatureExtractor> AnyFeatNonBin::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_anyFeatNonBin>(settings);
    }

    float DescriptorDistance_anyFeatureNonBin(const cv::Mat &a, const cv::Mat &b);
}

#endif //AF_VSLAM_FEATURE_ANYFEATNONBIN_H