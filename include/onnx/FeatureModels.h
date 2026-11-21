/**
 * Module: AllFeature-VSLAM - onnx/FeatureModels.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.1
 * - Created: 2026-10-16
 * - Updated: 2026-10-16 (static-shape exports: features_static_shapes)
 * - License: GPLv3 License
 *
 * The process-wide configuration of the learned feature models on ONNX Runtime and the sessions they share.
 *
 * Settings (the dev YAML, read by System before any extractor or matcher exists):
 *   features_models_dir    "feature_models"  the folder with superpoint_v1_<precision>.onnx, aliked_n16_<precision>.onnx,
 *                                            lightglue_{superpoint,aliked,sift}_<precision>.onnx and their sidecars
 *   features_provider      "auto"            ONNX Runtime execution provider for all of them: auto|cuda|coreml|cpu
 *   features_threads       4                 CPU-provider threads per session
 *   features_coreml_units  "ALL"             Core ML compute units
 *   features_static_shapes false             load the fixed-shape exports: the matchers' 'lightglue_<f>_L<L>_<precision>.onnx'
 *                                            padded to L keypoints per side (the smallest L that fits each pair; the
 *                                            dynamic file when none does; no padded file at all is a hard error), and,
 *                                            on the Core ML provider only, the detectors' '<stem>_<W>x<H>_<precision>.onnx'
 *                                            for the frame size (a missing one falls back to the dynamic file with a
 *                                            warning). Core ML compiles only bounded shapes, so this is the Mac's GPU
 *                                            path; on CUDA and the CPU the detectors load the dynamic file.
 *   features_superpoint_subpixel true        SuperPoint keeps only the peaks of its score map and places each at the
 *                                            score-weighted centre of the 3x3 scores around it (SuperPointOnnx::
 *                                            set_subpixel). false is upstream's selection at integer pixels.
 *                                            Its effect on accuracy is not established: the runs of 2026-11-21 (1
 *                                            or 2 per variant, see the settings file) are fewer than the run-to-run
 *                                            spread of that sequence requires.
 *   features_raco_covariance true            raco128 keypoints carry the 2x2 position covariance RaCo predicts
 *                                            (pixels^2) as their sigma2 / information; false: the fixed sigma of
 *                                            1 px the other learned features have. No difference in keyframe ATE
 *                                            on ETH table_3 over 5 runs each (medians 3.3 and 3.2 mm).
 *   features_precision     "fp32"            which export of every model is loaded, named in the file:
 *                                            "fp32" ('<stem>_fp32.onnx', exact against the PyTorch reference) or
 *                                            "fp16" ('<stem>_fp16.onnx': weights and internal tensors in half precision,
 *                                            float32 inputs and outputs; faster on the GPU, approximate).
 *                                            `pixi run setup` exports the files the settings ask for.
 *
 * Sharing: Tracking, LocalMapping and LoopClosing each own a FeatureMatcher; the LightGlue sessions are loaded once
 * per process here and handed out by variant (OrtSession::run serialises each session's runs). The extractors stay per
 * Tracking instance (one process has one).
 */

#ifndef AF_VSLAM_ONNX_FEATURE_MODELS_H
#define AF_VSLAM_ONNX_FEATURE_MODELS_H

#include <string>
#include <vector>

#include "onnx/LightGlueOnnx.h"
#include "onnx/OrtSession.h"

namespace AF_VSLAM::onnx
{

struct FeatureModelsConfig
{
    OrtSettings ort;                            // features_provider / features_threads / features_coreml_units
    std::string models_dir{"feature_models"};   // features_models_dir, resolved against the process cwd
    bool static_shapes{false};                  // features_static_shapes: the fixed-shape exports (Core ML)
    std::string precision{"fp32"};              // features_precision: "fp32" | "fp16", the export every model loads
    bool superpoint_subpixel{true};             // features_superpoint_subpixel: sub-pixel SuperPoint keypoints
    bool raco_covariance{true};                 // features_raco_covariance: RaCo's covariance as the keypoint's sigma2
};

/// The one configuration; System fills it from the settings YAML.
FeatureModelsConfig& feature_models_config();

/// `<models_dir>/<stem>_<precision>.onnx`, the precision being features_precision ("fp32" | "fp16").
/// @throws std::runtime_error if features_precision is neither "fp32" nor "fp16"
std::string feature_model_path(const std::string& stem);

/// `<models_dir>/<stem>_<W>x<H>_<precision>.onnx`: a detector's static export for one frame size.
/// If that file does not exist: the dynamic file feature_model_path(stem), with a warning naming the export command.
/// @param export_key  the model's `--only` name in tools/export_features.py ("superpoint" | "aliked"), for the warning
std::string static_model_path(const std::string& stem, int width, int height, const std::string& export_key);

/// The process-wide matcher for `features` ("superpoint" | "aliked" | "sift") for a pair with at most `n_max`
/// keypoints per side, created and warmed up on first use. With `static_shapes` on and `n_max` > 0: the smallest
/// padded export 'lightglue_<features>_L<L>_<precision>.onnx' in models_dir with L >= n_max, else (and with `n_max` 0) the
/// dynamic file - with one warning per variant the first time a pair outgrows every padded file.
/// @throws std::runtime_error if the model file or its sidecar is missing or inconsistent, or if static shapes are
/// on and no padded export of the variant exists at all.
LightGlueOnnx& shared_lightglue(const std::string& features, int n_max = 0);

/// Create every session of `features` now: the dynamic file and, with `static_shapes` on, every padded export in
/// models_dir. FeatureMatcher calls this at construction so that no session is created on the tracking thread later
/// (a Core ML session takes seconds to instantiate even from its compiled cache: a 9 s stall on the first pair that
/// outgrew the smallest padded file, measured on an M1 Pro, 2026-10-16). Returns the padded lengths found, ascending.
/// @throws std::runtime_error as shared_lightglue.
std::vector<int> preload_lightglue(const std::string& features);

} // namespace AF_VSLAM::onnx

#endif // AF_VSLAM_ONNX_FEATURE_MODELS_H
