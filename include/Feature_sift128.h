/**
 * Module: AllFeature-VSLAM - Feature_sift128.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 2.0 (cv::SIFT on the CPU; SiftGPU left in docs/plans/learned_features_onnxruntime.md step 6)
 * - Created: 2024-06-07
 * - Updated: 2026-10-16
 * - License: GPLv3 License
 *
 * sift128: SIFT keypoints + RootSIFT descriptors on OpenCV's CPU implementation (SiftCpu.h: the LightGlue
 * repository's SIFT extractor exactly - duplicate-orientation filter, RootSIFT, scale and orientation kept), matched
 * by LightGlue-SIFT (LIGHTGLUE_SIFT -> onnx::shared_lightglue("sift") in FeatureMatcher; the descriptor-only path is
 * the cross-checked brute-force L2 like the other float descriptors). The SLAM-facing octave and size stay 0 and 1
 * as before; the scale lives in cv::KeyPoint::size and the orientation in ::angle for the matcher.
 */

#ifndef AF_VSLAM_FEATURE_SIFT128_H
#define AF_VSLAM_FEATURE_SIFT128_H

#include <memory>

#include "Feature.h"
#include "FeatureExtractor.h"
#include "SiftCpu.h"

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
        explicit FeatureExtractor_sift128(std::shared_ptr<FeatureExtractorSettings> &settings_);
        ~FeatureExtractor_sift128() override = default;

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;

    private:
        std::unique_ptr<SiftCpu> sift_;
    };

    inline std::shared_ptr<FeatureExtractor> Sift128::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_sift128>(settings);
    }

    float DescriptorDistance_sift128(const cv::Mat &a, const cv::Mat &b);

}

#endif //AF_VSLAM_FEATURE_SIFT128_H
