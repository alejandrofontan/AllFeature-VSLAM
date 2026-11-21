/**
 * Module: AllFeature-VSLAM - onnx/RacoOnnx.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.2)
 * - Version: 1.0
 * - Created: 2026-11-21
 * - License: GPLv3 License
 *
 * RacoOnnx: RGB image -> RaCo keypoints (sub-pixel) with their ranker score and 2x2 position covariance, and an
 * ALIKED-n16 128-d descriptor at each, on ONNX Runtime. RaCo (Shenoi et al., 3DV 2026; github.com/cvg/RaCo) is a
 * detector only: its authors match it with ALIKED descriptors and a LightGlue trained for that pair, which the
 * exported graph (tools/export_features.py, RacoAlikedExport) and 'lightglue_raco' reproduce.
 *
 * The graph is the whole model: RaCo's normalisation, trunk and three heads; its own keypoint selection (softmax of
 * the score map over the image, 3x3 NMS, the K best by detection score - sidecar `topk` - and a sub-pixel offset);
 * score, ranker score and covariance sampled at each keypoint; ALIKED's trunk and descriptor head at those keypoints.
 * It returns K entries in detection-score order; one carrying score -1 is not a keypoint (an image with fewer than K
 * maxima) and detect() drops it. The host cuts the rest to its budget by ranker score.
 *
 * Conventions:
 *   keypoints    pixel centres at integers, the host's and OpenCV's (RaCo itself publishes x + 0.5, y + 0.5; the
 *                matcher's sidecar carries that offset, see LightGlueOnnx)
 *   response     cv::KeyPoint::response is the RANKER score (unbounded, higher is better): what a keypoint is
 *                chosen by. The detection probability is returned separately.
 *   covariances  (xx, xy, yy) in pixels^2 at the image's size
 *
 * Input contract (sidecar '<model>.onnx.yaml'): RGB in [0,1]. The host's images are BGR, so this class converts.
 *
 * Threading: per-call buffers only; OrtSession::run serialises the session's runs.
 */

#ifndef AF_VSLAM_ONNX_RACO_ONNX_H
#define AF_VSLAM_ONNX_RACO_ONNX_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "onnx/OrtSession.h"

namespace AF_VSLAM::onnx
{

class RacoOnnx
{
  public:
    /// @param onnx_path  'feature_models/raco_aliked_n16_fp32.onnx' or its '_fp16' twin
    /// @param sidecar_path  empty -> '<onnx_path>.yaml'
    RacoOnnx(const std::string& onnx_path, const OrtSettings& settings, const std::string& sidecar_path = "");

    /// Detect on a BGR (CV_8UC3; grey and BGRA are converted) image. One entry per keypoint in all four outputs:
    /// descriptors [N, 128] CV_32F; covariances (xx, xy, yy); detection_scores when asked for.
    void detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                std::vector<cv::Vec3f>& covariances, std::vector<float>* detection_scores = nullptr);

    [[nodiscard]] const std::string& provider() const { return session_.provider(); }
    [[nodiscard]] const std::string& model_path() const { return session_.model_path(); }
    [[nodiscard]] const std::string& precision() const { return precision_; }
    /// K of the graph's top-k: the most keypoints detect() can return.
    [[nodiscard]] int top_k() const { return top_k_; }
    [[nodiscard]] int descriptor_dim() const { return descriptor_dim_; }
    /// Wall time of the last detect(): the graph alone, and graph + conversions.
    [[nodiscard]] double last_graph_ms() const { return last_graph_ms_; }
    [[nodiscard]] double last_total_ms() const { return last_total_ms_; }

  private:
    void read_sidecar(const std::string& path);

    OrtSession session_;
    std::string input_name_;
    std::vector<std::string> output_names_;   // keypoints, descriptors, scores, ranker_scores, covariances
    std::string precision_{"fp32"};
    int top_k_{0};
    int descriptor_dim_{128};
    double last_graph_ms_{0.0};
    double last_total_ms_{0.0};
};

} // namespace AF_VSLAM::onnx

#endif // AF_VSLAM_ONNX_RACO_ONNX_H
