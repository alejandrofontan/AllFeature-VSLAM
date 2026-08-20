/**
 * Shared online-segmentation CLI plumbing (segmentation.md) for every entry
 * point (vslamlab_allfeature_mono / _rgbd / _mono_stream): the four
 * 'segmentation*' key:value arguments, the missing-model policy, and the
 * engine build/load announcements live here once so they cannot drift
 * between executables.
 */
#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <opencv2/core/core.hpp>

#include "afvslam_log.hpp"
#include "tensorrt_seg.hpp"

namespace segmentation {

struct SegmentationCLI
{
    static constexpr const char* kDefaultOnnx =
        "segmentation_models/efficientvit-seg-l1-ade20k_512x512.onnx";

    // ON by default. Pass 'segmentation:0' to disable, or
    // 'segmentation_onnx:<path>' to use another model.
    bool enabled{true};
    std::string onnx{kDefaultOnnx};
    std::string classes{};        // empty -> '<onnx>.classes.yaml'
    std::string precision{"fp16"};

    /// Consume one 'key:value' CLI argument; returns false if it isn't a
    /// segmentation key. "segmentation:" vs "segmentation_onnx:" et al. share
    /// a common stem, but the prefix match includes ':', so they cannot
    /// collide.
    bool parseArg(const std::string& arg)
    {
        std::string value;
        if (consume(arg, "segmentation", value)) {
            enabled = bool(std::stoi(value));
            AF_CONFIG_FIELD("Online segmentation: ", enabled);
            return true;
        }
        if (consume(arg, "segmentation_onnx", value)) {
            onnx = value;
            AF_CONFIG_FIELD("Online segmentation model: ", onnx);
            return true;
        }
        if (consume(arg, "segmentation_classes", value)) {
            classes = value;
            AF_CONFIG_FIELD("Online segmentation classes yaml: ", classes);
            return true;
        }
        if (consume(arg, "segmentation_precision", value)) {
            precision = value;
            AF_CONFIG_FIELD("Online segmentation precision: ", precision);
            return true;
        }
        return false;
    }

    /// Build/load the TensorRT engine (call before the SLAM threads exist).
    /// Returns nullptr - degrading the run to no-mask - when segmentation is
    /// disabled or the DEFAULT model file is missing (fresh clone, model not
    /// downloaded yet); an explicitly given path that is missing throws
    /// std::runtime_error instead.
    std::unique_ptr<TensorRTSeg> makeSegmenter() const
    {
        if (!enabled)
            return nullptr;
        if (!std::ifstream(onnx).good()) {
            if (onnx == kDefaultOnnx) {
                AF_WARN("Default segmentation model not found (" << onnx
                        << ") - running WITHOUT online segmentation; generate it with "
                        << "Thirdparty/Segmentation-TensorRT/convert2onnx/export_efficientvit_seg.py");
                return nullptr;
            }
            throw std::runtime_error("segmentation_onnx not found: '" + onnx + "'");
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
    static bool consume(const std::string& arg, const std::string& key, std::string& value)
    {
        const std::string prefix = key + ":";
        if (arg.rfind(prefix, 0) != 0)
            return false;
        value = arg.substr(prefix.size());
        return true;
    }
};

} // namespace segmentation
