/**
 * Module: AllFeature-VSLAM - onnx/LightGlueOnnx.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 */

#include "onnx/LightGlueOnnx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

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

/// One side's inputs: normalised keypoints [1,N,2], descriptors [1,N,D], and for SIFT scales [1,N] / oris [1,N].
struct SideBuffers
{
    std::vector<float> keypoints, descriptors, scales, oris;
};

void pack_side(const std::vector<cv::KeyPoint>& kps, const cv::Mat& desc, int input_dim, bool add_scale_ori,
               float shift_x, float shift_y, float scale, SideBuffers& b)
{
    const int n = static_cast<int>(kps.size());
    if (desc.rows != n || desc.cols != input_dim || desc.type() != CV_32F)
        throw std::runtime_error("LightGlueOnnx: descriptors must be [" + std::to_string(n) + ", " +
                                 std::to_string(input_dim) + "] CV_32F, got [" + std::to_string(desc.rows) + ", " +
                                 std::to_string(desc.cols) + "] type " + std::to_string(desc.type()));
    b.keypoints.resize(static_cast<size_t>(n) * 2);
    for (int i = 0; i < n; ++i)
    {
        b.keypoints[2 * i] = (kps[i].pt.x - shift_x) / scale;        // upstream normalize_keypoints
        b.keypoints[2 * i + 1] = (kps[i].pt.y - shift_y) / scale;
    }
    b.descriptors.resize(static_cast<size_t>(n) * input_dim);
    for (int i = 0; i < n; ++i)
        std::copy_n(desc.ptr<float>(i), input_dim, b.descriptors.data() + static_cast<size_t>(i) * input_dim);
    if (add_scale_ori)
    {
        b.scales.resize(n);
        b.oris.resize(n);
        for (int i = 0; i < n; ++i)
        {
            b.scales[i] = kps[i].size;                                        // px, as cv::SIFT reports it
            b.oris[i] = kps[i].angle * static_cast<float>(CV_PI / 180.0);     // upstream: np.deg2rad(k.angle)
        }
    }
}

} // namespace

LightGlueOnnx::LightGlueOnnx(const std::string& onnx_path, const OrtSettings& settings,
                             const std::string& sidecar_path)
    : session_(onnx_path, settings)
{
    read_sidecar(sidecar_path.empty() ? onnx_path + ".yaml" : sidecar_path);
    const size_t expected_inputs = add_scale_ori_ ? 8 : 4;
    const auto& out = session_.output_names();
    if (session_.input_names().size() != expected_inputs || out.size() != 1 || out[0] != output_name_)
        throw std::runtime_error("LightGlueOnnx: the sidecar's tensor names disagree with " + onnx_path + ":\n" +
                                 session_.describe_io());
    AF_INFO("LightGlueOnnx: " << onnx_path << " (" << features_ << ", " << input_dim_ << "-d"
                              << (add_scale_ori_ ? " + scale/ori" : "") << ", " << precision_
                              << " weights) on onnxruntime/" << session_.provider() << ", threshold "
                              << filter_threshold_);
}

void LightGlueOnnx::read_sidecar(const std::string& path)
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(path);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("LightGlueOnnx: cannot read sidecar " + path + ": " + e.what());
    }
    try
    {
        if (root["model"].as<std::string>() != "lightglue")
            throw std::runtime_error("model is '" + root["model"].as<std::string>() + "', not lightglue");
        features_ = root["features"].as<std::string>();
        input_dim_ = root["input_dim"].as<int>();
        add_scale_ori_ = root["add_scale_ori"].as<bool>();
        filter_threshold_ = root["filter_threshold"].as<float>();
        output_name_ = root["output_tensor"].as<std::string>();
        if (root["min_keypoints"])
            min_keypoints_ = root["min_keypoints"].as<int>();
        if (root["precision"])
            precision_ = root["precision"].as<std::string>();
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("LightGlueOnnx: sidecar " + path + ": " + e.what());
    }
}

std::vector<cv::DMatch> LightGlueOnnx::match(const std::vector<cv::KeyPoint>& keypoints0, const cv::Mat& descriptors0,
                                             const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& descriptors1,
                                             int image_width, int image_height)
{
    const auto t_start = Clock::now();
    std::vector<cv::DMatch> matches;
    const int n0 = static_cast<int>(keypoints0.size()), n1 = static_cast<int>(keypoints1.size());
    if (n0 < min_keypoints_ || n1 < min_keypoints_)
    {
        last_graph_ms_ = 0.0;
        last_total_ms_ = ms_since(t_start);
        return matches;
    }

    // upstream normalize_keypoints with image_size = [W, H]: shift = size / 2, scale = max(size) / 2
    const float shift_x = image_width / 2.0f, shift_y = image_height / 2.0f;
    const float scale = std::max(image_width, image_height) / 2.0f;
    SideBuffers s0, s1;
    pack_side(keypoints0, descriptors0, input_dim_, add_scale_ori_, shift_x, shift_y, scale, s0);
    pack_side(keypoints1, descriptors1, input_dim_, add_scale_ori_, shift_x, shift_y, scale, s1);

    std::vector<TensorView> inputs{
        {"keypoints_0", s0.keypoints.data(), {1, n0, 2}},
        {"descriptors_0", s0.descriptors.data(), {1, n0, input_dim_}},
        {"keypoints_1", s1.keypoints.data(), {1, n1, 2}},
        {"descriptors_1", s1.descriptors.data(), {1, n1, input_dim_}},
    };
    if (add_scale_ori_)
    {
        inputs.push_back({"scales_0", s0.scales.data(), {1, n0}});
        inputs.push_back({"oris_0", s0.oris.data(), {1, n0}});
        inputs.push_back({"scales_1", s1.scales.data(), {1, n1}});
        inputs.push_back({"oris_1", s1.oris.data(), {1, n1}});
    }
    const auto t_graph = Clock::now();
    const std::vector<Tensor> outputs = session_.run(inputs);
    last_graph_ms_ = ms_since(t_graph);

    // upstream filter_matches on the [N0+1, N1+1] log-assignment (dustbin row/column excluded from the argmax)
    const Tensor& la = outputs[0];
    if (la.shape.size() != 3 || la.dim(1) != n0 + 1 || la.dim(2) != n1 + 1)
        throw std::runtime_error("LightGlueOnnx: unexpected log_assignment shape");
    const int stride = n1 + 1;
    const float* S = la.data.data();
    std::vector<int> m0(n0, -1), m1(n1, -1);
    std::vector<float> max0(n0, -INFINITY), max1(n1, -INFINITY);
    for (int i = 0; i < n0; ++i)
    {
        const float* row = S + static_cast<size_t>(i) * stride;
        for (int j = 0; j < n1; ++j)
        {
            if (row[j] > max0[i])
            {
                max0[i] = row[j];
                m0[i] = j;
            }
            if (row[j] > max1[j])
            {
                max1[j] = row[j];
                m1[j] = i;
            }
        }
    }
    matches.reserve(std::min(n0, n1));
    for (int i = 0; i < n0; ++i)
    {
        const int j = m0[i];
        if (j < 0 || m1[j] != i)
            continue;                                   // not mutual
        const float score = std::exp(max0[i]);
        if (score > filter_threshold_)
            matches.emplace_back(i, j, 1.0f - score);
    }
    last_total_ms_ = ms_since(t_start);
    return matches;
}

} // namespace AF_VSLAM::onnx
