#ifndef ANYFEATURE_VSLAM_FEATURE_BRISK48_H
#define ANYFEATURE_VSLAM_FEATURE_BRISK48_H

#include "Feature.h"
#include "FeatureExtractor.h"
#include "brisk/brisk.h"

namespace ANYFEATURE_VSLAM {

    class Brisk48 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        const FeatureType  getType()           const override { return FEAT_BRISK; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
    private:
        inline static const std::string s_featureName    = "brisk48";
        inline static const Eigen::Matrix<float,3,1> s_color = {0, 122, 122};
    };

    class FeatureExtractor_brisk48 : public FeatureExtractor {
    public:

        cv::Ptr<cv::FeatureDetector> brisk_detector;
        cv::Ptr<cv::DescriptorExtractor> brisk_extractor;

        FeatureExtractor_brisk48(std::shared_ptr<FeatureExtractorSettings> &settings_);

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;
    };

    float DescriptorDistance_brisk48(const cv::Mat &a, const cv::Mat &b);

}

#endif //ANYFEATURE_VSLAM_FEATURE_BRISK48_H
