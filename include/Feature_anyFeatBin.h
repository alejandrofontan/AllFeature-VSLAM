#ifndef ANYFEATURE_VSLAM_FEATURE_ANYFEATBIN_H
#define ANYFEATURE_VSLAM_FEATURE_ANYFEATBIN_H

#include "Feature.h"
#include "FeatureExtractor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace ANYFEATURE_VSLAM {

    class FeatureExtractor_anyFeatBin;

    class AnyFeatBin : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_ANYFEATBIN; }
        const MatcherType getMatcherType()     const override { return BF_HAMMING; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float DescriptorDistance(const cv::Mat &a, const cv::Mat &b) const override { return 0.0f; };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;
    private:
        inline static const std::string s_featureName    = "anyfeatbin";
        inline static const std::string s_settingsYamlFile = "settings/anyFeatBin_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {200, 100, 50};
    };

    class FeatureExtractor_anyFeatBin : public FeatureExtractor {
    public:

        FeatureExtractor_anyFeatBin(std::shared_ptr<FeatureExtractorSettings> &settings_);

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;

    };

    inline std::shared_ptr<FeatureExtractor> AnyFeatBin::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_anyFeatBin>(settings);
    }

    float DescriptorDistance_anyFeatureBin(const cv::Mat &a, const cv::Mat &b);

}

#endif //ANYFEATURE_VSLAM_FEATURE_ANYFEATBIN_H