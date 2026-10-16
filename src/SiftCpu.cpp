/**
 * Module: AllFeature-VSLAM - SiftCpu.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 */

#include "SiftCpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

namespace AF_VSLAM
{

namespace
{

using Clock = std::chrono::steady_clock;

/// numpy's np.round: ties to even (std::nearbyint under the default rounding mode)
int round_half_even(float v)
{
    return static_cast<int>(std::nearbyint(v));
}

/// Upstream filter_dog_point with nms_radius 0: indices of the keypoints to keep. Two passes over the pixel each
/// keypoint rounds to: max response per pixel, then min |orientation| among the response winners.
std::vector<int> filter_dog_point(const std::vector<cv::KeyPoint>& kps)
{
    struct Cell
    {
        float max_response{-std::numeric_limits<float>::infinity()};
        float min_abs_angle{std::numeric_limits<float>::infinity()};
    };
    std::unordered_map<int64_t, Cell> cells;
    const auto key = [](const cv::KeyPoint& k) {
        const int64_t row = round_half_even(k.pt.y - 0.5f), col = round_half_even(k.pt.x - 0.5f);
        return (row << 32) ^ (col & 0xffffffffLL);
    };
    for (const auto& k : kps)
        cells[key(k)].max_response = std::max(cells[key(k)].max_response, k.response);
    std::vector<int> winners;
    for (int i = 0; i < static_cast<int>(kps.size()); ++i)
        if (kps[i].response == cells[key(kps[i])].max_response)
            winners.push_back(i);
    for (int i : winners)
    {
        Cell& c = cells[key(kps[i])];
        c.min_abs_angle = std::min(c.min_abs_angle, std::abs(kps[i].angle));
    }
    std::vector<int> keep;
    for (int i : winners)
        if (std::abs(kps[i].angle) == cells[key(kps[i])].min_abs_angle)
            keep.push_back(i);
    return keep;
}

/// Upstream sift_to_rootsift, float32 like torch: L1-normalise (eps 1e-6), clip at eps, sqrt, L2-normalise (eps).
void root_sift_in_place(float* d, int dim)
{
    constexpr float eps = 1e-6f;
    float l1 = 0.0f;
    for (int c = 0; c < dim; ++c)
        l1 += std::abs(d[c]);
    const float inv_l1 = 1.0f / std::max(l1, eps);
    float l2 = 0.0f;
    for (int c = 0; c < dim; ++c)
    {
        d[c] = std::sqrt(std::max(d[c] * inv_l1, eps));
        l2 += d[c] * d[c];
    }
    const float inv_l2 = 1.0f / std::max(std::sqrt(l2), eps);
    for (int c = 0; c < dim; ++c)
        d[c] *= inv_l2;
}

} // namespace

SiftCpu::SiftCpu(int max_keypoints, int n_octave_layers, double contrast_threshold, double edge_threshold,
                 double sigma)
    : sift_(cv::SIFT::create(max_keypoints, n_octave_layers, contrast_threshold, edge_threshold, sigma))
{
}

void SiftCpu::detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    const auto t_start = Clock::now();
    cv::Mat grey;
    if (image.channels() == 1)
        grey = image;
    else
        cv::cvtColor(image, grey, image.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    if (grey.type() != CV_8UC1)
        throw std::runtime_error("SiftCpu: expected an 8-bit image");

    std::vector<cv::KeyPoint> all;
    cv::Mat all_desc;
    sift_->detectAndCompute(grey, cv::noArray(), all, all_desc);

    const std::vector<int> keep = filter_dog_point(all);
    const int n = static_cast<int>(keep.size());
    keypoints.clear();
    keypoints.reserve(n);
    descriptors.create(n, all_desc.cols, CV_32F);
    for (int i = 0; i < n; ++i)
    {
        keypoints.push_back(all[keep[i]]);
        keypoints.back().class_id = i;
        all_desc.row(keep[i]).copyTo(descriptors.row(i));
        root_sift_in_place(descriptors.ptr<float>(i), descriptors.cols);
    }
    last_total_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
}

} // namespace AF_VSLAM
