#ifndef AF_VSLAM_FEATURE_superpoint256_H
#define AF_VSLAM_FEATURE_superpoint256_H

#include "Feature.h"
#include "FeatureExtractor.h"

#ifdef CHECK
#undef CHECK
#endif

#include "Thirdparty/SuperPoint-LightGlue-TensorRT/include/super_point.h"

namespace AF_VSLAM {

    class FeatureExtractor_superpoint256;

    class Superpoint256 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        FeatureType  getType()           const override { return FEAT_SUPERPOINT256; }
        MatcherType getMatcherType()     const override { return LIGHTGLUE_SUPERPOINT; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override{ return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2); };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;

    private:
        inline static const std::string s_featureName    = "superpoint256";
        inline static const std::string s_settingsYamlFile = "settings/superpoint256_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {122, 0, 255};
    };

    class FeatureExtractor_superpoint256 : public FeatureExtractor {
    public:

        std::shared_ptr<SuperPoint> extractor;

        FeatureExtractor_superpoint256(std::shared_ptr<FeatureExtractorSettings> &settings_);
        ~FeatureExtractor_superpoint256() {}

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;
    };

    inline std::shared_ptr<FeatureExtractor> Superpoint256::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_superpoint256>(settings);
    }

    float DescriptorDistance_superpoint256(const cv::Mat &a, const cv::Mat &b);

}

#endif //AF_VSLAM_FEATURE_superpoint256_H
