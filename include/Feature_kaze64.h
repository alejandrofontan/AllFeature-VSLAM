#ifndef ANYFEATURE_VSLAM_FEATURE_KAZE64_H
#define ANYFEATURE_VSLAM_FEATURE_KAZE64_H

#include "Feature.h"
#include "FeatureExtractor.h"

namespace ANYFEATURE_VSLAM {

    class Kaze64 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_KAZE64; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
    private:
        inline static const std::string s_featureName    = "kaze64";
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

    float DescriptorDistance_kaze64(const cv::Mat &a, const cv::Mat &b);
}

#endif //ANYFEATURE_VSLAM_FEATURE_KAZE64_H
