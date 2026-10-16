/**
 * Module: AllFeature-VSLAM - onnx/FeatureModels.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 */

#include "onnx/FeatureModels.h"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "afvslam_log.hpp"

namespace AF_VSLAM::onnx
{

FeatureModelsConfig& feature_models_config()
{
    static FeatureModelsConfig config;
    return config;
}

std::string feature_model_path(const std::string& file)
{
    return feature_models_config().models_dir + "/" + file;
}

LightGlueOnnx& shared_lightglue(const std::string& features)
{
    static std::mutex mutex;
    static std::map<std::string, std::unique_ptr<LightGlueOnnx>> matchers;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = matchers.find(features);
    if (it == matchers.end())
    {
        auto matcher = std::make_unique<LightGlueOnnx>(feature_model_path("lightglue_" + features + ".onnx"),
                                                       feature_models_config().ort);
        // Warm-up: the first Run of a session pays allocation and (CUDA) kernel selection; keep it out of the
        // first frame. Sixteen points per side on a 640x480 frame, descriptors of the variant's size.
        std::vector<cv::KeyPoint> kps;
        for (int i = 0; i < 16; ++i)
            kps.emplace_back(40.0f * i + 20.0f, 30.0f * (i % 4) + 15.0f, 3.0f, 10.0f * i);
        cv::Mat desc(16, matcher->input_dim(), CV_32F);
        cv::randn(desc, 0.0, 1.0);
        for (int i = 0; i < desc.rows; ++i)
            cv::normalize(desc.row(i), desc.row(i));
        matcher->match(kps, desc, kps, desc, 640, 480);
        AF_INFO("LightGlue (" << features << ") warmed up: " << matcher->last_graph_ms() << " ms for 16x16 points");
        it = matchers.emplace(features, std::move(matcher)).first;
    }
    return *it->second;
}

} // namespace AF_VSLAM::onnx
