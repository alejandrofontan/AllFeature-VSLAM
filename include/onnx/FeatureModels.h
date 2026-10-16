/**
 * Module: AllFeature-VSLAM - onnx/FeatureModels.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.1
 * - Created: 2026-10-16
 * - Updated: 2026-10-16 (static-shape exports: features_static_shapes)
 * - License: GPLv3 License
 *
 * The process-wide configuration of the learned feature models on ONNX Runtime and the sessions they share
 * (docs/plans/learned_features_onnxruntime.md, steps 5-7; decisions Q1, Q3).
 *
 * Settings (the dev YAML, read by System before any extractor or matcher exists):
 *   features_models_dir    "feature_models"  the folder with superpoint_v1.onnx, aliked_n16.onnx,
 *                                            lightglue_{superpoint,aliked,sift}.onnx and their sidecars
 *   features_provider      "auto"            ONNX Runtime execution provider for all of them: auto|cuda|coreml|cpu
 *   features_threads       4                 CPU-provider threads per session
 *   features_coreml_units  "ALL"             Core ML compute units
 *   features_static_shapes false             load the fixed-shape exports instead of the dynamic files: the detectors'
 *                                            '<stem>_<W>x<H>.onnx' for the frame size (loaded on the first frame), the
 *                                            matchers' 'lightglue_<f>_L<L>.onnx' padded to L keypoints per side (the
 *                                            smallest L that fits each pair; the dynamic file when none does). Core ML
 *                                            compiles only bounded shapes, so this is the Mac's GPU path (step 7);
 *                                            a missing file is a hard error naming the export command.
 * The model files are the fp32 exports by name; their fp16 twins are approximate (the host note) and get a key only
 * if a provider measurement asks for them.
 *
 * Sharing: Tracking, LocalMapping and LoopClosing each own a FeatureMatcher; the LightGlue sessions are loaded once
 * per process here and handed out by variant (OrtSession::run is concurrency-safe). The extractors stay per
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
    bool static_shapes{false};                  // features_static_shapes: the fixed-shape exports (Core ML)
};

/// The one configuration; System fills it from the settings YAML.
FeatureModelsConfig& feature_models_config();

/// `<models_dir>/<file>`.
std::string feature_model_path(const std::string& file);

/// `<models_dir>/<stem>_<W>x<H>.onnx` for `file` = '<stem>.onnx': a detector's static export for one frame size.
/// @param export_key  the model's `--only` name in tools/export_features.py ("superpoint" | "aliked"), for the error
/// @throws std::runtime_error naming the export command if the file does not exist.
std::string static_model_path(const std::string& file, int width, int height, const std::string& export_key);

/// The process-wide matcher for `features` ("superpoint" | "aliked" | "sift") for a pair with at most `n_max`
/// keypoints per side, created and warmed up on first use. With `static_shapes` on and `n_max` > 0: the smallest
/// padded export 'lightglue_<features>_L<L>.onnx' in models_dir with L >= n_max, else (and with `n_max` 0) the
/// dynamic file - with one warning per variant the first time a pair outgrows every padded file.
/// @throws std::runtime_error if the model file or its sidecar is missing or inconsistent, or if static shapes are
/// on and no padded export of the variant exists at all.
LightGlueOnnx& shared_lightglue(const std::string& features, int n_max = 0);

} // namespace AF_VSLAM::onnx

#endif // AF_VSLAM_ONNX_FEATURE_MODELS_H
