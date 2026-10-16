/**
 * Module: AllFeature-VSLAM - onnx/SuperPointOnnx.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 */

#include "onnx/SuperPointOnnx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "afvslam_log.hpp"

namespace AF_VSLAM::onnx
{

namespace
{

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// Upstream `sample_descriptors(keypoints, descriptors, s=8)` for one keypoint: the (x, y) pixel position is
/// mapped into the 1/8-resolution descriptor map with SuperPoint's own convention, then sampled bilinearly with
/// align_corners=True and zero padding (torch grid_sample defaults), then L2-normalised.
void sample_descriptor(const float* desc_map, int dim, int map_h, int map_w, float x, float y, float* out)
{
    constexpr float s = 8.0f;
    // keypoints - s/2 + 0.5, divided by (w*s - s/2 - 0.5, h*s - s/2 - 0.5), times 2 minus 1  -> [-1, 1]
    const float u = (x - s / 2 + 0.5f) / (map_w * s - s / 2 - 0.5f) * 2.0f - 1.0f;
    const float v = (y - s / 2 + 0.5f) / (map_h * s - s / 2 - 0.5f) * 2.0f - 1.0f;
    // align_corners=True: -1 -> 0, +1 -> size - 1
    const float ix = (u + 1.0f) / 2.0f * static_cast<float>(map_w - 1);
    const float iy = (v + 1.0f) / 2.0f * static_cast<float>(map_h - 1);
    const int x0 = static_cast<int>(std::floor(ix)), y0 = static_cast<int>(std::floor(iy));
    const int x1 = x0 + 1, y1 = y0 + 1;
    const float wx1 = ix - static_cast<float>(x0), wy1 = iy - static_cast<float>(y0);
    const float wx0 = 1.0f - wx1, wy0 = 1.0f - wy1;
    const auto inside = [&](int xx, int yy) { return xx >= 0 && xx < map_w && yy >= 0 && yy < map_h; };
    const float w00 = inside(x0, y0) ? wx0 * wy0 : 0.0f, w10 = inside(x1, y0) ? wx1 * wy0 : 0.0f;
    const float w01 = inside(x0, y1) ? wx0 * wy1 : 0.0f, w11 = inside(x1, y1) ? wx1 * wy1 : 0.0f;
    const size_t plane = static_cast<size_t>(map_h) * map_w;
    double norm2 = 0.0;
    for (int c = 0; c < dim; ++c)
    {
        const float* p = desc_map + static_cast<size_t>(c) * plane;
        float value = 0.0f;
        if (w00 != 0.0f) value += w00 * p[y0 * map_w + x0];
        if (w10 != 0.0f) value += w10 * p[y0 * map_w + x1];
        if (w01 != 0.0f) value += w01 * p[y1 * map_w + x0];
        if (w11 != 0.0f) value += w11 * p[y1 * map_w + x1];
        out[c] = value;
        norm2 += static_cast<double>(value) * value;
    }
    const float inv = norm2 > 0.0 ? static_cast<float>(1.0 / std::sqrt(norm2)) : 0.0f;
    for (int c = 0; c < dim; ++c)
        out[c] *= inv;
}

/// Upstream `simple_nms(scores, radius)` on a [H,W] float map: a pixel survives if it equals the max of its
/// (2r+1)^2 neighbourhood (cv::dilate with a rectangular kernel == max_pool2d with padding r), then two suppression
/// passes: pixels within r of a survivor are suppressed, and a non-suppressed pixel that is the max of the suppressed
/// map joins the survivors. Non-survivors become 0. Exact mirror of the Python specification (superpoint_tail).
void simple_nms(cv::Mat& scores, int radius)
{
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * radius + 1, 2 * radius + 1));
    cv::Mat pooled, max_mask, supp_mask, supp_scores, pooled_supp, new_max_mask, keep;
    cv::dilate(scores, pooled, kernel);
    cv::compare(scores, pooled, max_mask, cv::CMP_EQ);                 // 255 where scores == max_pool(scores)
    for (int it = 0; it < 2; ++it)
    {
        cv::dilate(max_mask, supp_mask, kernel);                       // supp_mask = max_pool(max_mask) > 0
        scores.copyTo(supp_scores);
        supp_scores.setTo(0.0f, supp_mask);                            // where(supp_mask, 0, scores)
        cv::dilate(supp_scores, pooled_supp, kernel);
        cv::compare(supp_scores, pooled_supp, new_max_mask, cv::CMP_EQ);
        cv::bitwise_and(new_max_mask, ~supp_mask, keep);               // new_max_mask & ~supp_mask
        cv::bitwise_or(max_mask, keep, max_mask);
    }
    cv::Mat zeros = cv::Mat::zeros(scores.size(), scores.type());
    zeros.copyTo(scores, ~max_mask);                                   // where(max_mask, scores, 0)
}

} // namespace

SuperPointOnnx::SuperPointOnnx(const std::string& onnx_path, const OrtSettings& settings,
                               const std::string& sidecar_path)
    : session_(onnx_path, settings)
{
    read_sidecar(sidecar_path.empty() ? onnx_path + ".yaml" : sidecar_path);
    const auto& in = session_.input_names();
    const auto& out = session_.output_names();
    if (in.size() != 1 || in[0] != input_name_ || out.size() != 2 || out[0] != scores_name_ ||
        out[1] != descriptors_name_)
        throw std::runtime_error("SuperPointOnnx: the sidecar's tensor names disagree with " + onnx_path + ":\n" +
                                 session_.describe_io());
    AF_INFO("SuperPointOnnx: " << onnx_path << " (" << precision_ << " weights) on onnxruntime/"
                               << session_.provider() << ", threshold " << detection_threshold_);
}

void SuperPointOnnx::read_sidecar(const std::string& path)
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(path);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("SuperPointOnnx: cannot read sidecar " + path + ": " + e.what());
    }
    try
    {
        if (root["model"].as<std::string>() != "superpoint")
            throw std::runtime_error("model is '" + root["model"].as<std::string>() + "', not superpoint");
        input_name_ = root["input_tensor"].as<std::string>();
        scores_name_ = root["output_tensors"][0].as<std::string>();
        descriptors_name_ = root["output_tensors"][1].as<std::string>();
        detection_threshold_ = root["detection_threshold"].as<float>();
        descriptor_dim_ = root["descriptor_dim"].as<int>();
        nms_radius_ = root["nms_radius"].as<int>();
        remove_borders_ = root["remove_borders"].as<int>();
        if (root["precision"])
            precision_ = root["precision"].as<std::string>();
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("SuperPointOnnx: sidecar " + path + ": " + e.what());
    }
}

void SuperPointOnnx::detect(const cv::Mat& image, int max_keypoints, std::vector<cv::KeyPoint>& keypoints,
                            cv::Mat& descriptors)
{
    const auto t_start = Clock::now();
    cv::Mat grey;
    if (image.channels() == 1)
        grey = image;
    else
        cv::cvtColor(image, grey, image.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    if (grey.type() != CV_8UC1)
        throw std::runtime_error("SuperPointOnnx: expected an 8-bit image");

    // [1,1,H,W] float32 in [0,1]
    const int H = grey.rows, W = grey.cols;
    std::vector<float> input(static_cast<size_t>(H) * W);
    for (int r = 0; r < H; ++r)
    {
        const uchar* row = grey.ptr<uchar>(r);
        float* dst = input.data() + static_cast<size_t>(r) * W;
        for (int c = 0; c < W; ++c)
            dst[c] = static_cast<float>(row[c]) * (1.0f / 255.0f);
    }
    const auto t_graph = Clock::now();
    const std::vector<Tensor> outputs = session_.run({{input_name_, input.data(), {1, 1, H, W}}});
    last_graph_ms_ = ms_since(t_graph);

    const Tensor& scores = outputs[0];        // [1, H8, W8]
    const Tensor& desc_map = outputs[1];      // [1, dim, H8/8, W8/8]
    const int h8 = static_cast<int>(scores.dim(1)), w8 = static_cast<int>(scores.dim(2));
    const int dim = static_cast<int>(desc_map.dim(1));
    const int map_h = static_cast<int>(desc_map.dim(2)), map_w = static_cast<int>(desc_map.dim(3));
    if (dim != descriptor_dim_)
        throw std::runtime_error("SuperPointOnnx: descriptor map has " + std::to_string(dim) + " channels, sidecar says " +
                                 std::to_string(descriptor_dim_));

    // Tail, upstream order: NMS -> border mask -> threshold -> best K by score -> (x, y) -> sample -> L2
    cv::Mat score_map(h8, w8, CV_32F, const_cast<float*>(scores.data.data()));
    simple_nms(score_map, nms_radius_);
    struct Candidate
    {
        float score;
        int x, y;
    };
    std::vector<Candidate> candidates;
    const int b = remove_borders_;
    for (int y = b; y < h8 - b; ++y)                                   // border pixels are -1 upstream: never above threshold
    {
        const float* row = score_map.ptr<float>(y);
        for (int x = b; x < w8 - b; ++x)
            if (row[x] > detection_threshold_)
                candidates.push_back({row[x], x, y});
    }
    if (max_keypoints > 0 && static_cast<int>(candidates.size()) > max_keypoints)
    {
        std::partial_sort(candidates.begin(), candidates.begin() + max_keypoints, candidates.end(),
                          [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
        candidates.resize(max_keypoints);
    }

    const int n = static_cast<int>(candidates.size());
    keypoints.clear();
    keypoints.reserve(n);
    descriptors.create(n, dim, CV_32F);
    for (int i = 0; i < n; ++i)
    {
        const Candidate& c = candidates[i];
        cv::KeyPoint kp;
        kp.pt = cv::Point2f(static_cast<float>(c.x), static_cast<float>(c.y));
        kp.response = c.score;
        kp.size = 1.0f;
        kp.angle = 0.0f;
        kp.octave = 0;
        kp.class_id = i;
        keypoints.push_back(kp);
        sample_descriptor(desc_map.data.data(), dim, map_h, map_w, kp.pt.x, kp.pt.y, descriptors.ptr<float>(i));
    }
    last_total_ms_ = ms_since(t_start);
}

} // namespace AF_VSLAM::onnx
