#ifndef ANYFEATURE_VSLAM_FEATURE_SIFT128_H
#define ANYFEATURE_VSLAM_FEATURE_SIFT128_H

#include "Feature.h"
#include "FeatureExtractor.h"
#include "SiftGPU.h"

namespace ANYFEATURE_VSLAM {

    class Sift128 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_SIFT128; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
    private:
        inline static const std::string s_featureName    = "sift128";
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

    float DescriptorDistance_sift128(const cv::Mat &a, const cv::Mat &b);
}

#endif //ANYFEATURE_VSLAM_FEATURE_SIFT128_H
