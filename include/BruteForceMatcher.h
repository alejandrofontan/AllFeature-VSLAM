/**
 * BruteForceMatcher.h — SIMD/OpenMP brute-force descriptor matching.
 *
 * Drop-in replacements for the cv::BFMatcher(<norm>, crossCheck=true) members previously
 * used by FeatureMatcher (issue #13 addendum P2). Semantics replicated exactly:
 * mutual-nearest-neighbour ("cross-check") 1-NN matching with OpenCV's first-minimum
 * tie-breaking in both directions, results in ascending query order.
 *
 * Hamming is bit-exact w.r.t. cv::BFMatcher. L2 selects by squared distance (monotone,
 * same ordering) with Eigen-vectorised accumulation, so distances can differ from
 * OpenCV's in the last ulp — see test_bfmatcher_parity for the empirical check.
 */

#ifndef AF_VSLAM_BRUTEFORCEMATCHER_H
#define AF_VSLAM_BRUTEFORCEMATCHER_H

#include <opencv2/core.hpp>
#include <vector>

namespace AF_VSLAM
{

// query/train: CV_8U descriptor matrices, one row per descriptor (any byte width).
std::vector<cv::DMatch> bruteforce_match_hamming(const cv::Mat& query, const cv::Mat& train);

// query/train: CV_32F descriptor matrices, one row per descriptor.
std::vector<cv::DMatch> bruteforce_match_l2(const cv::Mat& query, const cv::Mat& train);

} // namespace AF_VSLAM

#endif // AF_VSLAM_BRUTEFORCEMATCHER_H
