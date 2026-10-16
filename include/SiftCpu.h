/**
 * Module: AllFeature-VSLAM - SiftCpu.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 *
 * SiftCpu: grey image -> SIFT keypoints with scale and orientation + RootSIFT descriptors, on OpenCV's CPU
 * implementation (docs/plans/learned_features_onnxruntime.md, decision Q7: cv::SIFT everywhere, SiftGPU leaves).
 * It mirrors the LightGlue repository's SIFT extractor (lightglue/sift.py, OpenCV backend) exactly, because the
 * LightGlue-SIFT weights were trained on that output and the C++ parity reference is that code on the same
 * library:
 *
 *   1. cv::SIFT(nfeatures = max_keypoints, nOctaveLayers = 4, contrastThreshold = 0.0066667 (COLMAP's),
 *      edgeThreshold = 10, sigma = 1.6), detectAndCompute on the 8-bit grey image;
 *   2. the duplicate filter `filter_dog_point`: OpenCV emits several keypoints at one pixel (one per dominant
 *      orientation); keep, per pixel (round(x - 0.5), round(y - 0.5), ties-to-even like numpy), the one with the
 *      highest response, and among equal responses the smallest |orientation|;
 *   3. RootSIFT (`sift_to_rootsift`): L1-normalise, clip at 1e-6, square root, L2-normalise.
 *
 * Scale (cv::KeyPoint::size, px) and orientation (cv::KeyPoint::angle, degrees) are kept on the keypoints: the
 * LightGlue-SIFT matcher takes them as inputs (LightGlueOnnx converts the angle to radians). Today's SiftGPU
 * extractor zeroed both and the libtorch matcher ran with the ALIKED weights; this class is the fix by
 * construction (plan Q8).
 */

#ifndef AF_VSLAM_SIFT_CPU_H
#define AF_VSLAM_SIFT_CPU_H

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace AF_VSLAM
{

class SiftCpu
{
  public:
    /// @param max_keypoints  cv::SIFT's nfeatures: the strongest keypoints by contrast are kept before the
    ///                       duplicate filter (0 = all)
    explicit SiftCpu(int max_keypoints, int n_octave_layers = 4, double contrast_threshold = 0.0066667,
                     double edge_threshold = 10.0, double sigma = 1.6);

    /// Detect on a grey (CV_8UC1; BGR is converted) image. Descriptors: [N, 128] CV_32F RootSIFT.
    void detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    /// Wall time of the last detect().
    [[nodiscard]] double last_total_ms() const { return last_total_ms_; }

  private:
    cv::Ptr<cv::SIFT> sift_;
    double last_total_ms_{0.0};
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_SIFT_CPU_H
