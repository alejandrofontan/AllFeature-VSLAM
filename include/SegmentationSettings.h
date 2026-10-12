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

// The segmentation engine follows the CMake backend option ALLFEATURE_WITH_TENSORRT (CMakeLists.txt): with
// it, `Segmenter` is the TensorRT engine of Thirdparty/Segmentation-TensorRT; without it, a stub that is
// never constructed (makeSegmenter() returns nullptr, or throws when segmentation was requested
// explicitly), so the entry points compile and link unchanged on a GPU-free build.
#ifdef ALLFEATURE_WITH_TENSORRT
#include "tensorrt_seg.hpp"
#endif

namespace segmentation {

#ifdef ALLFEATURE_WITH_TENSORRT
using Segmenter = TensorRTSeg;
#else
class Segmenter
{
  public:
    cv::Mat inferMask(const cv::Mat&) { throw std::logic_error("segmentation: no backend compiled in"); }
};
#endif

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
    std::unique_ptr<Segmenter> makeSegmenter() const
    {
        if (method == "none")
            return nullptr;
#ifndef ALLFEATURE_WITH_TENSORRT
        if (method_explicit || onnx_explicit)
            throw std::runtime_error(
                "segmentation: " + method + " is not available in this build: it needs the CMake option "
                "ALLFEATURE_WITH_TENSORRT=ON (docs/plans/mac_port.md). Set segmentation: none or rebuild with that backend.");
        AF_WARN("Online segmentation not compiled in (ALLFEATURE_WITH_TENSORRT=OFF) - running WITHOUT online segmentation");
        return nullptr;
#else
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
#endif
    }

private:
    // Whether the value came from the YAML (explicit) or is the compiled-in
    // default - decides hard-error vs degrade-with-warning on a missing model.
    bool method_explicit{false};
    bool onnx_explicit{false};
};

} // namespace segmentation
