/**
 * Module: AllFeature-VSLAM - onnx/SuperPointOnnx.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 *
 * SuperPointOnnx: grey image -> SuperPoint keypoints + 256-d descriptors on ONNX Runtime. The exported graph
 * (tools/export_features.py, SuperPointDense; plan Q4) ends at the dense maps: a score per pixel after softmax,
 * pixel shuffle, NMS (radius 4) and border masking, and the L2-normalised descriptor map at 1/8 resolution. This
 * class runs that graph and then the tail that `superpoint_tail` in the export tool specifies, in upstream's
 * order: keep scores > detection_threshold, the best `max_keypoints` by score, bilinear descriptor sampling with
 * SuperPoint's s=8 normalisation and align_corners, L2-normalise. The sidecar '<model>.onnx.yaml' supplies the
 * tensor names, threshold, descriptor size and the file's weight precision.
 *
 * Any image size: the three max-pools floor it, so the maps cover the top-left (W//8*8) x (H//8*8) pixels and no
 * keypoint can land in the last W%8 columns or H%8 rows. Keypoint coordinates are integer pixel positions in the
 * image frame (upstream SuperPoint has no sub-pixel refinement); the score goes to cv::KeyPoint::response.
 *
 * Threading: detect() uses a per-call input buffer and OrtSession::run is concurrency-safe, so one instance may
 * serve several threads; the host's extractor runs on the tracking thread only.
 */

#ifndef AF_VSLAM_ONNX_SUPERPOINT_ONNX_H
#define AF_VSLAM_ONNX_SUPERPOINT_ONNX_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "onnx/OrtSession.h"

namespace AF_VSLAM::onnx
{

class SuperPointOnnx
{
  public:
    /// @param onnx_path  'feature_models/superpoint_v1.onnx' or its '_fp16' twin
    /// @param sidecar_path  empty -> '<onnx_path>.yaml'
    SuperPointOnnx(const std::string& onnx_path, const OrtSettings& settings, const std::string& sidecar_path = "");

    /// Detect on a grey (CV_8UC1; BGR is converted) image. `max_keypoints` <= 0 keeps every point above the
    /// threshold. Descriptors: [N, 256] CV_32F, L2-normalised, one row per keypoint.
    void detect(const cv::Mat& image, int max_keypoints, std::vector<cv::KeyPoint>& keypoints,
                cv::Mat& descriptors);

    [[nodiscard]] const std::string& provider() const { return session_.provider(); }
    [[nodiscard]] const std::string& model_path() const { return session_.model_path(); }
    [[nodiscard]] const std::string& precision() const { return precision_; }
    [[nodiscard]] float detection_threshold() const { return detection_threshold_; }
    [[nodiscard]] int descriptor_dim() const { return descriptor_dim_; }
    /// Wall time of the last detect(): the graph alone, and graph + tail.
    [[nodiscard]] double last_graph_ms() const { return last_graph_ms_; }
    [[nodiscard]] double last_total_ms() const { return last_total_ms_; }

  private:
    void read_sidecar(const std::string& path);

    OrtSession session_;
    std::string input_name_;
    std::string scores_name_;
    std::string descriptors_name_;
    std::string precision_{"fp32"};
    float detection_threshold_{0.0005f};
    int descriptor_dim_{256};
    double last_graph_ms_{0.0};
    double last_total_ms_{0.0};
};

} // namespace AF_VSLAM::onnx

#endif // AF_VSLAM_ONNX_SUPERPOINT_ONNX_H
