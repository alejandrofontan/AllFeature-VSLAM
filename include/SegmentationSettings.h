/**
 * Shared online-segmentation settings (segmentation.md) for every entry point
 * (vslamlab_allfeature_mono / _rgbd / _mono_stream): backend selection, the
 * missing-model policy, and the engine build/load announcements live here once
 * so they cannot drift between executables.
 *
 * Resolved from the settings YAML (keys all optional), mirroring the VPR
 * resolution in System.cc: an explicit request that cannot be satisfied is a
 * hard error, a defaulted one degrades to "none" with a warning.
 *
 *   segmentation:           "efficientvit" (default) | "none"
 *   segmentation_onnx:      model path (default: kDefaultOnnx below)
 *   segmentation_classes:   classes yaml (default: '<onnx>.classes.yaml')
 *   segmentation_precision: "fp16" (default)
 */
#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <opencv2/core/core.hpp>
#include <yaml-cpp/yaml.h>

#include "afvslam_log.hpp"
#include "tensorrt_seg.hpp"

namespace segmentation {

struct SegmentationSettings
{
    static constexpr const char* kDefaultOnnx =
        "segmentation_models/efficientvit-seg-l1-ade20k_512x512.onnx";

    std::string method{"efficientvit"};
    std::string onnx{kDefaultOnnx};
    std::string classes{};        // empty -> '<onnx>.classes.yaml'
    std::string precision{"fp16"};

    /// Read the segmentation keys from the (already loaded) settings YAML.
    /// Throws std::runtime_error on an unknown backend name.
    void load(const YAML::Node& settings)
    {
        if (settings["segmentation"]) {
            method = settings["segmentation"].as<std::string>();
            method_explicit = true;
        }
        if (settings["segmentation_onnx"]) {
            onnx = settings["segmentation_onnx"].as<std::string>();
            onnx_explicit = true;
        }
        if (settings["segmentation_classes"])
            classes = settings["segmentation_classes"].as<std::string>();
        if (settings["segmentation_precision"])
            precision = settings["segmentation_precision"].as<std::string>();

        if (method != "none" && method != "efficientvit")
            throw std::runtime_error("Unknown segmentation backend '" + method
                                     + "' (options: efficientvit, none)");
        AF_CONFIG_FIELD("Online segmentation: ", method);
    }

    /// Build/load the TensorRT engine (call before the SLAM threads exist).
    /// Returns nullptr - no masks - when segmentation resolves to "none":
    /// set explicitly, or defaulted while the DEFAULT model file is missing
    /// (fresh checkout before the model download) - the latter with a warning.
    /// An explicit request (segmentation: efficientvit, or an explicit
    /// segmentation_onnx path) whose model is missing throws instead.
    std::unique_ptr<TensorRTSeg> makeSegmenter() const
    {
        if (method == "none")
            return nullptr;
        if (!std::ifstream(onnx).good()) {
            if (onnx_explicit)
                throw std::runtime_error("segmentation_onnx not found: '" + onnx + "'");
            if (method_explicit)
                throw std::runtime_error(
                    "segmentation: efficientvit requested but the model is missing ('" + onnx
                    + "') - let the VSLAM-LAB wrapper download it, generate it with "
                      "Thirdparty/Segmentation-TensorRT/convert2onnx/export_efficientvit_seg.py, "
                      "or set segmentation: none");
            AF_WARN("Default segmentation model not found (" << onnx
                    << ") - running WITHOUT online segmentation; generate it with "
                    << "Thirdparty/Segmentation-TensorRT/convert2onnx/export_efficientvit_seg.py");
            return nullptr;
        }

        // stdout is fully buffered when redirected to a log file, so flush
        // explicitly or these lines stay invisible during the ~1 min build.
        const bool engineCached =
            std::ifstream(onnx + "." + precision + ".engine").good();
        AF_INFO("Online segmentation: "
                << (engineCached ? "loading cached TensorRT engine for "
                                 : "building TensorRT engine (~1 min) for ")
                << onnx << " ...");
        std::cout.flush();

        auto segmenter = std::make_unique<TensorRTSeg>(onnx, precision, classes);
        // Absorb the one-time lazy CUDA/TRT warmup (~70 ms) here rather than
        // on frame 0.
        segmenter->inferMask(cv::Mat::zeros(480, 640, CV_8UC1));
        AF_INFO("Online segmentation ready (engine "
                << (segmenter->loadedFromCache() ? "cached" : "built") << ": "
                << segmenter->enginePath() << ")");
        std::cout.flush();
        return segmenter;
    }

private:
    // Whether the value came from the YAML (explicit) or is the compiled-in
    // default - decides hard-error vs degrade-with-warning on a missing model.
    bool method_explicit{false};
    bool onnx_explicit{false};
};

} // namespace segmentation
