/**
 * Module: AllFeature-VSLAM - onnx/RacoOnnx.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.2)
 * - Version: 1.0
 * - Created: 2026-11-21
 * - License: GPLv3 License
 */

#include "onnx/RacoOnnx.h"

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

RacoOnnx::RacoOnnx(const std::string& onnx_path, const OrtSettings& settings, const std::string& sidecar_path)
    : session_(onnx_path, settings)
{
    read_sidecar(sidecar_path.empty() ? onnx_path + ".yaml" : sidecar_path);
    const auto& in = session_.input_names();
    if (in.size() != 1 || in[0] != input_name_ || session_.output_names() != output_names_)
        throw std::runtime_error("RacoOnnx: the sidecar's tensor names disagree with " + onnx_path + ":\n" +
                                 session_.describe_io());
    AF_INFO("RacoOnnx: " << onnx_path << " (" << precision_ << " weights) on onnxruntime/" << session_.provider()
                         << ", top-" << top_k_ << " keypoints by detection score, each with a ranker score and a covariance");
}

void RacoOnnx::read_sidecar(const std::string& path)
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(path);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("RacoOnnx: cannot read sidecar " + path + ": " + e.what());
    }
    try
    {
        if (root["model"].as<std::string>() != "raco")
            throw std::runtime_error("model is '" + root["model"].as<std::string>() + "', not raco");
        input_name_ = root["input_tensor"].as<std::string>();
        output_names_ = root["output_tensors"].as<std::vector<std::string>>();
        if (output_names_.size() != 5)
            throw std::runtime_error("output_tensors must name five tensors: keypoints, descriptors, scores, "
                                     "ranker_scores, covariances");
        descriptor_dim_ = root["descriptor_dim"].as<int>();
        top_k_ = root["topk"].as<int>();
        if (top_k_ <= 0)
            throw std::runtime_error("topk must be positive, got " + std::to_string(top_k_));
        if (root["precision"])
            precision_ = root["precision"].as<std::string>();
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("RacoOnnx: sidecar " + path + ": " + e.what());
    }
}

void RacoOnnx::detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                      std::vector<cv::Vec3f>& covariances, std::vector<float>* detection_scores)
{
    const auto t_start = Clock::now();
    cv::Mat rgb;
    switch (image.channels())
    {
        case 3: cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB); break;
        case 1: cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB); break;
        case 4: cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB); break;
        default: throw std::runtime_error("RacoOnnx: unsupported number of channels");
    }
    if (rgb.type() != CV_8UC3)
        throw std::runtime_error("RacoOnnx: expected an 8-bit image");

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

    const Tensor& kp = outputs[0];       // [K, 2]
    const Tensor& desc = outputs[1];     // [K, dim]
    const Tensor& score = outputs[2];    // [K]
    const Tensor& ranker = outputs[3];   // [K]
    const Tensor& cov = outputs[4];      // [K, 3]
    const int n = static_cast<int>(kp.dim(0));
    const int dim = static_cast<int>(desc.dim(1));
    if (dim != descriptor_dim_)
        throw std::runtime_error("RacoOnnx: descriptors are " + std::to_string(dim) + "-d, sidecar says " +
                                 std::to_string(descriptor_dim_));
    if (cov.shape.size() != 2 || cov.dim(0) != n || cov.dim(1) != 3 || score.data.size() != static_cast<size_t>(n) ||
        ranker.data.size() != static_cast<size_t>(n))
        throw std::runtime_error("RacoOnnx: unexpected output shapes from " + session_.model_path());

    // K entries in detection-score order, the ones that are not maxima at all marked with score -1: a prefix is valid
    int n_keep = 0;
    for (int i = 0; i < n; ++i)
        if (score.data[i] >= 0.0f)
            n_keep = i + 1;
    keypoints.clear();
    keypoints.reserve(n_keep);
    covariances.clear();
    covariances.reserve(n_keep);
    if (detection_scores)
        detection_scores->assign(score.data.begin(), score.data.begin() + n_keep);
    descriptors.create(n_keep, dim, CV_32F);
    for (int i = 0; i < n_keep; ++i)
    {
        cv::KeyPoint k;
        k.pt = cv::Point2f(kp.data[2 * i], kp.data[2 * i + 1]);
        k.response = ranker.data[i];   // the ranker score: what the host selects by
        k.size = 1.0f;
        k.angle = 0.0f;
        k.octave = 0;
        k.class_id = i;
        keypoints.push_back(k);
        covariances.emplace_back(cov.data[3 * i], cov.data[3 * i + 1], cov.data[3 * i + 2]);
        std::copy_n(desc.data.data() + static_cast<size_t>(i) * dim, dim, descriptors.ptr<float>(i));
    }
    last_total_ms_ = ms_since(t_start);
}

} // namespace AF_VSLAM::onnx
