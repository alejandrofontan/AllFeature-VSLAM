#ifndef ANYFEATURE_VSLAM_FEATURE_ALIKED128_H
#define ANYFEATURE_VSLAM_FEATURE_ALIKED128_H

#include "Feature.h"
#include "FeatureExtractor.h"
#include "feature/ALIKED.hpp"

namespace ANYFEATURE_VSLAM {

    class Aliked128 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_ALIKED128; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
    private:
        inline static const std::string s_featureName    = "aliked128";
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

    float DescriptorDistance_aliked128(const cv::Mat &a, const cv::Mat &b);

}

#endif //ANYFEATURE_VSLAM_FEATURE_ALIKED128_H
