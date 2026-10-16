/**
 * Module: AllFeature-VSLAM - onnx/LightGlueOnnx.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 *
 * LightGlueOnnx: two keypoint sets with descriptors -> matches, on ONNX Runtime. One class for the three exported
 * matchers (lightglue_superpoint / lightglue_aliked / lightglue_sift.onnx, tools/export_features.py
 * LightGlueExport): which one it is comes from the sidecar '<model>.onnx.yaml' - the descriptor size the input
 * projection expects (`input_dim`), whether the positional encoding takes scale and orientation as well
 * (`add_scale_ori`, SIFT), the match threshold and the tensor names - never from code, so a fourth variant is a
 * file (plan §4b).
 *
 * The graph is the fixed-depth matcher: input projection, positional encoding, 9 transformer layers, the last
 * assignment head; it returns the [N0+1, N1+1] log-assignment with the dustbin row and column. This class does
 * what upstream does around it: normalises the keypoints with the real image size ((xy - size/2) / (max(W,H)/2),
 * the fix of the 320x240 bug of the TensorRT path), takes scale (cv::KeyPoint::size, px) and orientation
 * (cv::KeyPoint::angle, degrees -> radians) for the SIFT variant, and extracts the matches with upstream's
 * `filter_matches`: mutual row/column argmax over the [N0, N1] block and exp(score) > filter_threshold. Fewer than
 * two keypoints on a side (sidecar `min_keypoints`) gives no matches: the graph has no empty-set branch.
 *
 * Padded files ('lightglue_<features>_L<L>.onnx', sidecar `padded_length` L; tools/export_features.py
 * LightGluePaddedExport, the Core ML lever of step 7: Core ML compiles only bounded shapes): both sides are padded
 * to exactly L rows (zeros) and two more inputs, mask_0 / mask_1 [1,L], carry 0 for a real row and the sidecar's
 * `mask_padded_value` for a padded one - the additive attention bias that keeps real keypoints from attending to
 * the padding. The output is [L+1, L+1]; match() reads the [N0, N1] block with row stride L+1. A pair with more
 * than L keypoints on a side does not fit: the caller (onnx::shared_lightglue) routes it to the dynamic file.
 *
 * Output: cv::DMatch with queryIdx in set 0, trainIdx in set 1 and distance = 1 - matching score (the host's
 * convention for learned matchers).
 *
 * Threading: per-call buffers only; OrtSession::run is concurrency-safe, so one instance serves the tracking and
 * the local-mapping threads.
 */

#ifndef AF_VSLAM_ONNX_LIGHTGLUE_ONNX_H
#define AF_VSLAM_ONNX_LIGHTGLUE_ONNX_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "onnx/OrtSession.h"

namespace AF_VSLAM::onnx
{

class LightGlueOnnx
{
  public:
    /// @param onnx_path  'feature_models/lightglue_<features>.onnx' or its '_fp16' twin
    /// @param sidecar_path  empty -> '<onnx_path>.yaml'
    LightGlueOnnx(const std::string& onnx_path, const OrtSettings& settings, const std::string& sidecar_path = "");

    /// Match set 0 against set 1. Descriptors: [N, input_dim] CV_32F, one row per keypoint, as the extractor
    /// produced them. `image_width/height`: the frame size both sets live in (keypoint normalisation).
    std::vector<cv::DMatch> match(const std::vector<cv::KeyPoint>& keypoints0, const cv::Mat& descriptors0,
                                  const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& descriptors1,
                                  int image_width, int image_height);

    [[nodiscard]] const std::string& features() const { return features_; }
    [[nodiscard]] int input_dim() const { return input_dim_; }
    [[nodiscard]] bool add_scale_ori() const { return add_scale_ori_; }
    [[nodiscard]] float filter_threshold() const { return filter_threshold_; }
    /// 0 for the dynamic file; L for a padded file, which takes at most L keypoints per side.
    [[nodiscard]] int padded_length() const { return padded_length_; }
    [[nodiscard]] const std::string& provider() const { return session_.provider(); }
    [[nodiscard]] const std::string& model_path() const { return session_.model_path(); }
    [[nodiscard]] const std::string& precision() const { return precision_; }
    /// Wall time of the last match(): the graph alone, and graph + packing + match extraction.
    [[nodiscard]] double last_graph_ms() const { return last_graph_ms_; }
    [[nodiscard]] double last_total_ms() const { return last_total_ms_; }

  private:
    void read_sidecar(const std::string& path);

    OrtSession session_;
    std::string features_;
    int input_dim_{256};
    bool add_scale_ori_{false};
    float filter_threshold_{0.1f};
    int min_keypoints_{2};
    int padded_length_{0};
    float mask_padded_value_{-1e9f};
    std::string precision_{"fp32"};
    std::string output_name_;
    double last_graph_ms_{0.0};
    double last_total_ms_{0.0};
};

} // namespace AF_VSLAM::onnx

#endif // AF_VSLAM_ONNX_LIGHTGLUE_ONNX_H
