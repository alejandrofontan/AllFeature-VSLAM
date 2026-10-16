/**
 * Module: AllFeature-VSLAM - onnx/AlikedOnnx.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 *
 * AlikedOnnx: RGB image -> ALIKED-n16 keypoints (sub-pixel) + 128-d descriptors, on ONNX Runtime. The exported
 * graph (tools/export_features.py, AlikedExport) is the whole model: replicate-padding to a multiple of 32 and
 * the crop back, the trunk with its two deformable-convolution blocks (ONNX DeformConv, native on the CPU and
 * CUDA providers), the DKD detector head in threshold mode (NMS radius 2, score > 0.2, soft-argmax refinement)
 * and the SDDH descriptor head. The dynamic export returns keypoints [N,2] in pixels of the input frame,
 * descriptors [N,128] L2-normalised and scores [N], with N data-dependent (every maximum above the threshold); the
 * static exports for Core ML ('<stem>_<W>x<H>.onnx', sidecar `topk`) return exactly K by score and detect() keeps
 * the ones above the threshold - the same set whenever fewer than K pass. The host applies its own budget afterwards.
 *
 * Input contract (sidecar '<model>.onnx.yaml'): RGB in [0,1]. The host's images are BGR (cv::imread), so this
 * class converts - the libtorch path handed ALIKED the BGR image as RGB (plan Q8, fixed by construction).
 *
 * Threading: per-call buffers only; OrtSession::run serialises the session's runs.
 */

#ifndef AF_VSLAM_ONNX_ALIKED_ONNX_H
#define AF_VSLAM_ONNX_ALIKED_ONNX_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "onnx/OrtSession.h"

namespace AF_VSLAM::onnx
{

class AlikedOnnx
{
  public:
    /// @param onnx_path  'feature_models/aliked_n16.onnx' or its '_fp16' twin
    /// @param sidecar_path  empty -> '<onnx_path>.yaml'
    AlikedOnnx(const std::string& onnx_path, const OrtSettings& settings, const std::string& sidecar_path = "");

    /// Detect on a BGR (CV_8UC3; grey and BGRA are converted) image. Descriptors: [N, 128] CV_32F, one row per
    /// keypoint; the score goes to cv::KeyPoint::response.
    void detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    [[nodiscard]] const std::string& provider() const { return session_.provider(); }
    [[nodiscard]] const std::string& model_path() const { return session_.model_path(); }
    [[nodiscard]] const std::string& precision() const { return precision_; }
    [[nodiscard]] float detection_threshold() const { return detection_threshold_; }
    [[nodiscard]] int descriptor_dim() const { return descriptor_dim_; }
    /// Wall time of the last detect(): the graph alone, and graph + conversions.
    [[nodiscard]] double last_graph_ms() const { return last_graph_ms_; }
    [[nodiscard]] double last_total_ms() const { return last_total_ms_; }

  private:
    void read_sidecar(const std::string& path);

    OrtSession session_;
    std::string input_name_;
    std::string keypoints_name_, descriptors_name_, scores_name_;
    std::string precision_{"fp32"};
    float detection_threshold_{0.2f};
    int top_k_{0};   // > 0: the graph returns exactly top_k_ keypoints by score (static exports); detect() thresholds
    int descriptor_dim_{128};
    double last_graph_ms_{0.0};
    double last_total_ms_{0.0};
};

} // namespace AF_VSLAM::onnx

#endif // AF_VSLAM_ONNX_ALIKED_ONNX_H
