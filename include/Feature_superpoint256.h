/**
 * Module: AllFeature-VSLAM - Feature_superpoint256.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 2.0 (ONNX Runtime; the TensorRT engine left in docs/plans/learned_features_onnxruntime.md step 6)
 * - Created: 2024-06-07
 * - Updated: 2026-10-16
 * - License: GPLv3 License
 *
 * superpoint256: SuperPoint keypoints + 256-d descriptors on ONNX Runtime (onnx/SuperPointOnnx.h), matched by
 * LightGlue-SuperPoint (LIGHTGLUE_SUPERPOINT -> onnx::shared_lightglue("superpoint") in FeatureMatcher). The model
 * file is `<features_models_dir>/superpoint_v1.onnx` (exported locally by `pixi run export-superpoint`: the Magic
 * Leap weights are not redistributable, plan Q6); the runtime provider comes from the features_* settings.
 */

#ifndef AF_VSLAM_FEATURE_superpoint256_H
#define AF_VSLAM_FEATURE_superpoint256_H

#include <memory>

#include "Feature.h"
#include "FeatureExtractor.h"
#include "onnx/SuperPointOnnx.h"

namespace AF_VSLAM {

    class FeatureExtractor_superpoint256;

    class Superpoint256 : public Feature {
    public:
        const std::string& getFeatureName()    const override { return s_featureName; }
        FeatureType  getType()           const override { return FEAT_SUPERPOINT256; }
        MatcherType getMatcherType()     const override { return LIGHTGLUE_SUPERPOINT; }
        const Eigen::Matrix<float,3,1>& getColor() const override { return s_color; }
        const std::string& getSettingsYamlFile() const override { return s_settingsYamlFile; }
        float descriptor_distance(const cv::Mat &a, const cv::Mat &b) const override{ return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2); };
        std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const override;

        /// The model file inside features_models_dir (fp32 export of tools/export_features.py)
        inline static const std::string kModelFile = "superpoint_v1.onnx";

    private:
        inline static const std::string s_featureName    = "superpoint256";
        inline static const std::string s_settingsYamlFile = "settings/superpoint256_settings.yaml";
        inline static const Eigen::Matrix<float,3,1> s_color = {122, 0, 255};
    };

    class FeatureExtractor_superpoint256 : public FeatureExtractor {
    public:
        explicit FeatureExtractor_superpoint256(std::shared_ptr<FeatureExtractorSettings> &settings_);
        ~FeatureExtractor_superpoint256() override = default;

        void detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

        [[nodiscard]] int GetKeypointOctave(const cv::KeyPoint &keypoint) const override;
        [[nodiscard]] float GetKeypointSize(const cv::KeyPoint &keypoint) const override;

    private:
        std::unique_ptr<onnx::SuperPointOnnx> extractor_;
    };

    inline std::shared_ptr<FeatureExtractor> Superpoint256::createExtractor(
        std::shared_ptr<FeatureExtractorSettings> settings) const {
        return std::make_shared<FeatureExtractor_superpoint256>(settings);
    }

    float DescriptorDistance_superpoint256(const cv::Mat &a, const cv::Mat &b);

}

#endif //AF_VSLAM_FEATURE_superpoint256_H
