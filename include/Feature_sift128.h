#ifndef AF_VSLAM_FEATURE_SIFT128_H
#define AF_VSLAM_FEATURE_SIFT128_H

#include "Feature.h"
#include "FeatureExtractor.h"
#include "SiftGPU.h"

namespace AF_VSLAM {

    class FeatureExtractor_sift128;

    class Sift128 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        FeatureType  getType()           const override { return FEAT_SIFT128; }
        MatcherType getMatcherType()     const override { return LIGHTGLUE_SIFT; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_L2); };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;
    private:
        inline static const std::string s_featureName    = "sift128";
        inline static const std::string s_settingsYamlFile = "settings/sift128_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {255, 0, 0};
    };

    class FeatureExtractor_sift128 : public FeatureExtractor {
    public:

        std::shared_ptr<SiftGPU> sift;

        FeatureExtractor_sift128(std::shared_ptr<FeatureExtractorSettings> &settings_);
        ~FeatureExtractor_sift128() {
            sift.reset();
        }

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        void detectKeypoints(std::vector<cv::KeyPoint> &keypoints, const Image &img,
                             const float &detectTh, const int &nOctaves) const;

        void computeDescriptors(cv::Mat &descriptors, const Image &img) const;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;
    };

    inline std::shared_ptr<FeatureExtractor> Sift128::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_sift128>(settings);
    }

    float DescriptorDistance_sift128(const cv::Mat &a, const cv::Mat &b);
}

#endif //AF_VSLAM_FEATURE_SIFT128_H
