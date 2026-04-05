#ifndef ANYFEATURE_VSLAM_FEATURE_superpoint256_H
#define ANYFEATURE_VSLAM_FEATURE_superpoint256_H

#include "Feature.h"
#include "FeatureExtractor.h"

#ifdef CHECK
#undef CHECK
#endif

#include "Thirdparty/SuperPoint-LightGlue-TensorRT/include/super_point.h"

namespace ANYFEATURE_VSLAM {

    class Superpoint256 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_SUPERPOINT256; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }

    private:
        inline static const std::string s_featureName    = "superpoint256";
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

    float DescriptorDistance_superpoint256(const cv::Mat &a, const cv::Mat &b);

}

#endif //ANYFEATURE_VSLAM_FEATURE_superpoint256_H
