#ifndef AF_VSLAM_FEATURE_SURF64_H
#define AF_VSLAM_FEATURE_SURF64_H

#include "Feature.h"
#include "FeatureExtractor.h"
//#include "opencv2/xfeatures2d/cudafeatures2d.hpp"
#include <opencv2/xfeatures2d/cuda.hpp>

namespace AF_VSLAM {

    class FeatureExtractor_surf64;

    class Surf64 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        FeatureType  getType()           const override { return FEAT_SURF64; }
        MatcherType getMatcherType()     const override { return BF_L2; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_L2); };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;
    private:
        inline static const std::string s_featureName    = "surf64";
        inline static const std::string s_settingsYamlFile = "settings/surf64_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {0, 0, 255};
    };

    class FeatureExtractor_surf64 : public FeatureExtractor {
    public:

        std::shared_ptr<cv::cuda::SURF_CUDA> surf;

        FeatureExtractor_surf64(std::shared_ptr<FeatureExtractorSettings> &settings_);

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;
    };

    inline std::shared_ptr<FeatureExtractor> Surf64::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_surf64>(settings);
    }

    float DescriptorDistance_surf64(const cv::Mat &a, const cv::Mat &b);
}


#endif //AF_VSLAM_FEATURE_SURF64_H
