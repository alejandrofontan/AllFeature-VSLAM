#ifndef AF_VSLAM_FEATURE_ORB32_H
#define AF_VSLAM_FEATURE_ORB32_H

#include "Feature.h"
#include "FeatureExtractor.h"

namespace AF_VSLAM {

    class FeatureExtractor_orb32;

    class Orb32 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        FeatureType  getType()           const override { return FEAT_ORB32; }
        MatcherType getMatcherType()     const override { return BF_HAMMING; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }

        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override;
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;
    private:
        inline static const std::string s_featureName    = "orb32";
        inline static const std::string s_settingsYamlFile = "settings/orb32_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {129, 149, 251};
    };

    class FeatureExtractor_orb32 : public FeatureExtractor {
    public:

        const int PATCH_SIZE = 31;
        const int HALF_PATCH_SIZE = 15;
        const int EDGE_THRESHOLD = 19;
        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;
        std::vector<cv::Mat> mvImagePyramid;
        int iniThFAST{20};
        int minThFAST{7};
        std::vector<int> mnFeaturesPerLevel;
        std::vector<int> umax;
        std::vector<cv::Point> pattern;

        FeatureExtractor_orb32(std::shared_ptr<FeatureExtractorSettings> &settings_);

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;

        void ComputePyramid(cv::Mat image);
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    };

    inline std::shared_ptr<FeatureExtractor> Orb32::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_orb32>(settings);
    }

    float DescriptorDistance_orb32(const cv::Mat &a, const cv::Mat &b);
}

#endif //AF_VSLAM_FEATURE_ORB32_H
