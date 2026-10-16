/**
 * Module: AllFeature-VSLAM - onnx/FeatureModels.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 *
 * The process-wide configuration of the learned feature models on ONNX Runtime and the sessions they share
 * (docs/plans/learned_features_onnxruntime.md, steps 5-6; decisions Q1, Q3).
 *
 * Settings (the dev YAML, read by System before any extractor or matcher exists):
 *   features_models_dir    "feature_models"  the folder with superpoint_v1.onnx, aliked_n16.onnx,
 *                                            lightglue_{superpoint,aliked,sift}.onnx and their sidecars
 *   features_provider      "auto"            ONNX Runtime execution provider for all of them: auto|cuda|coreml|cpu
 *   features_threads       4                 CPU-provider threads per session
 *   features_coreml_units  "ALL"             Core ML compute units
 * The model files are the fp32 exports by name; their fp16 twins are approximate (the host note) and get a key only
 * if a provider measurement asks for them.
 *
 * Sharing: Tracking, LocalMapping and LoopClosing each own a FeatureMatcher; the three LightGlue variants are loaded
 * once per process here and handed out by name (OrtSession::run is concurrency-safe). The extractors stay per
 * Tracking instance (one process has one).
 */

#ifndef AF_VSLAM_ONNX_FEATURE_MODELS_H
#define AF_VSLAM_ONNX_FEATURE_MODELS_H

#include <string>

#include "onnx/LightGlueOnnx.h"
#include "onnx/OrtSession.h"

namespace AF_VSLAM::onnx
{

struct FeatureModelsConfig
{
    OrtSettings ort;                            // features_provider / features_threads / features_coreml_units
    std::string models_dir{"feature_models"};   // features_models_dir, resolved against the process cwd
};

/// The one configuration; System fills it from the settings YAML.
FeatureModelsConfig& feature_models_config();

/// `<models_dir>/<file>`.
std::string feature_model_path(const std::string& file);

/// The process-wide matcher for `features` ("superpoint" | "aliked" | "sift"), created and warmed up on first use.
/// @throws std::runtime_error if the model file or its sidecar is missing or inconsistent.
LightGlueOnnx& shared_lightglue(const std::string& features);

} // namespace AF_VSLAM::onnx

#endif // AF_VSLAM_ONNX_FEATURE_MODELS_H
