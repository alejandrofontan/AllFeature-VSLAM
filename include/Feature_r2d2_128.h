#ifndef ANYFEATURE_VSLAM_FEATURE_R2D2_128_H
#define ANYFEATURE_VSLAM_FEATURE_R2D2_128_H

#include "Feature.h"
#include "FeatureExtractor.h"

namespace ANYFEATURE_VSLAM {

    class R2d2_128 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_R2D2; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
    private:
        inline static const std::string s_featureName    = "r2d2_128";
        inline static const Eigen::Matrix<float,3,1> s_color = {255 , 0, 255};
    };

    class FeatureExtractor_r2d2_128 : public FeatureExtractor {
    public:
        FeatureExtractor_r2d2_128(std::shared_ptr<FeatureExtractorSettings> &settings_);

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;
    };

    float DescriptorDistance_r2d2_128(const cv::Mat &a, const cv::Mat &b);
}

#endif //ANYFEATURE_VSLAM_FEATURE_R2D2_128_H
