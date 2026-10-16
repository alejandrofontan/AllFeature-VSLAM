/**
 * Module: AllFeature-VSLAM - onnx/AlikedOnnx.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 */

#include "onnx/AlikedOnnx.h"

#include <chrono>
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

} // namespace

AlikedOnnx::AlikedOnnx(const std::string& onnx_path, const OrtSettings& settings, const std::string& sidecar_path)
    : session_(onnx_path, settings)
{
    read_sidecar(sidecar_path.empty() ? onnx_path + ".yaml" : sidecar_path);
    const auto& in = session_.input_names();
    const auto& out = session_.output_names();
    if (in.size() != 1 || in[0] != input_name_ || out.size() != 3 || out[0] != keypoints_name_ ||
        out[1] != descriptors_name_ || out[2] != scores_name_)
        throw std::runtime_error("AlikedOnnx: the sidecar's tensor names disagree with " + onnx_path + ":\n" +
                                 session_.describe_io());
    AF_INFO("AlikedOnnx: " << onnx_path << " (" << precision_ << " weights) on onnxruntime/" << session_.provider()
                           << ", threshold " << detection_threshold_);
}

void AlikedOnnx::read_sidecar(const std::string& path)
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(path);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("AlikedOnnx: cannot read sidecar " + path + ": " + e.what());
    }
    try
    {
        if (root["model"].as<std::string>() != "aliked")
            throw std::runtime_error("model is '" + root["model"].as<std::string>() + "', not aliked");
        input_name_ = root["input_tensor"].as<std::string>();
        keypoints_name_ = root["output_tensors"][0].as<std::string>();
        descriptors_name_ = root["output_tensors"][1].as<std::string>();
        scores_name_ = root["output_tensors"][2].as<std::string>();
        detection_threshold_ = root["detection_threshold"].as<float>();
        descriptor_dim_ = root["descriptor_dim"].as<int>();
        if (root["precision"])
            precision_ = root["precision"].as<std::string>();
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("AlikedOnnx: sidecar " + path + ": " + e.what());
    }
}

void AlikedOnnx::detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    const auto t_start = Clock::now();
    cv::Mat rgb;
    switch (image.channels())
    {
        case 3: cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB); break;
        case 1: cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB); break;
        case 4: cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB); break;
        default: throw std::runtime_error("AlikedOnnx: unsupported number of channels");
    }
    if (rgb.type() != CV_8UC3)
        throw std::runtime_error("AlikedOnnx: expected an 8-bit image");

    // [1,3,H,W] float32 in [0,1], planar RGB
    const int H = rgb.rows, W = rgb.cols;
    const size_t plane = static_cast<size_t>(H) * W;
    std::vector<float> input(3 * plane);
    for (int r = 0; r < H; ++r)
    {
        const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(r);
        const size_t base = static_cast<size_t>(r) * W;
        for (int c = 0; c < W; ++c)
            for (int ch = 0; ch < 3; ++ch)
                input[ch * plane + base + c] = static_cast<float>(row[c][ch]) * (1.0f / 255.0f);
    }
    const auto t_graph = Clock::now();
    const std::vector<Tensor> outputs = session_.run({{input_name_, input.data(), {1, 3, H, W}}});
    last_graph_ms_ = ms_since(t_graph);

    const Tensor& kp = outputs[0];     // [N, 2]
    const Tensor& desc = outputs[1];   // [N, dim]
    const Tensor& sc = outputs[2];     // [N]
    const int n = static_cast<int>(kp.dim(0));
    const int dim = static_cast<int>(desc.dim(1));
    if (dim != descriptor_dim_)
        throw std::runtime_error("AlikedOnnx: descriptors are " + std::to_string(dim) + "-d, sidecar says " +
                                 std::to_string(descriptor_dim_));
    keypoints.clear();
    keypoints.reserve(n);
    descriptors.create(n, dim, CV_32F);
    for (int i = 0; i < n; ++i)
    {
        cv::KeyPoint k;
        k.pt = cv::Point2f(kp.data[2 * i], kp.data[2 * i + 1]);
        k.response = sc.data[i];
        k.size = 1.0f;
        k.angle = 0.0f;
        k.octave = 0;
        k.class_id = i;
        keypoints.push_back(k);
        std::copy_n(desc.data.data() + static_cast<size_t>(i) * dim, dim, descriptors.ptr<float>(i));
    }
    last_total_ms_ = ms_since(t_start);
}

} // namespace AF_VSLAM::onnx
