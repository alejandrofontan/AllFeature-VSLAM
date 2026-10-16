/**
 * Module: AllFeature-VSLAM - Feature_aliked128.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 2.0 (ONNX Runtime; the libtorch port left in docs/plans/learned_features_onnxruntime.md step 6)
 * - Created: 2024-06-07
 * - Updated: 2026-10-16
 * - License: GPLv3 License
 *
 * aliked128: ALIKED-n16 sub-pixel keypoints + 128-d descriptors on ONNX Runtime (onnx/AlikedOnnx.h), matched by
 * LightGlue-ALIKED (LIGHTGLUE_ALIKED -> onnx::shared_lightglue("aliked") in FeatureMatcher). The model file is
 * `<features_models_dir>/aliked_n16.onnx` (`pixi run download-feature-models`). The graph detects in threshold
 * mode; the host keeps its octree distribution to the keypoint budget, as before.
 */

#ifndef AF_VSLAM_FEATURE_ALIKED128_H
#define AF_VSLAM_FEATURE_ALIKED128_H

#include <memory>

#include "Feature.h"
#include "FeatureExtractor.h"
#include "onnx/AlikedOnnx.h"

namespace AF_VSLAM {

    class FeatureExtractor_aliked128;

    class Aliked128 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        FeatureType  getType()           const override { return FEAT_ALIKED128; }
        MatcherType getMatcherType()     const override { return LIGHTGLUE_ALIKED; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override { return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2); };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;

        /// The model file inside features_models_dir (fp32 export of tools/export_features.py)
        inline static const std::string kModelFile = "aliked_n16.onnx";

    private:
        inline static const std::string s_featureName    = "aliked128";
        inline static const std::string s_settingsYamlFile = "settings/aliked128_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {181, 243, 249};
    };

    class FeatureExtractor_aliked128 : public FeatureExtractor {
    public:
        explicit FeatureExtractor_aliked128(std::shared_ptr<FeatureExtractorSettings> &settings_);
        ~FeatureExtractor_aliked128() override = default;

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;

    private:
        std::unique_ptr<onnx::AlikedOnnx> extractor_;
    };

    inline std::shared_ptr<FeatureExtractor> Aliked128::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_aliked128>(settings);
    }

    float DescriptorDistance_aliked128(const cv::Mat &a, const cv::Mat &b);

}

#endif //AF_VSLAM_FEATURE_ALIKED128_H
