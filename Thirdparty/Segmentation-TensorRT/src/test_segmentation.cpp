/**
 * Module: AllFeature-VSLAM - Segmentation-TensorRT - test_segmentation.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 0.3 (full inference: preprocess -> enqueueV3 -> binary mask)
 * - Created: 2026-08-17
 * - License: GPLv3 License
 *
 * Standalone test harness for the online segmentation module (segmentation.md,
 * Phase 1). Runs TensorRTSeg on one image and writes 'trt_mask.png' (x255 for
 * visibility) and 'trt_overlay.png' (dynamic region tinted red) to the current
 * directory - directly comparable with the *_selfcheck_*.png files produced by
 * convert2onnx/export_efficientvit_seg.py --test_image (the Python reference).
 *
 * Usage: test_segmentation <model.onnx> <image> [fp16|fp32]
 */

#include <chrono>
#include <cstdio>

#include <opencv2/imgcodecs.hpp>

#include "tensorrt_seg.hpp"

namespace
{
long msSince(const std::chrono::steady_clock::time_point& t0)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                 t0)
        .count();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <model.onnx> <image> [fp16|fp32]\n", argv[0]);
        return 1;
    }
    const std::string onnxPath = argv[1];
    const std::string imagePath = argv[2];
    const std::string precision = argc > 3 ? argv[3] : "fp16";

    const cv::Mat frame = cv::imread(imagePath);
    if (frame.empty())
    {
        std::printf("cannot read image %s\n", imagePath.c_str());
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    segmentation::TensorRTSeg seg(onnxPath, precision);
    std::printf("engine %s in %ld ms: %s\nIO tensors:\n%s",
                seg.loadedFromCache() ? "loaded" : "built", msSince(t0),
                seg.enginePath().c_str(), seg.describeIOTensors().c_str());

    // First call includes one-time lazy CUDA/TRT warmup; report it separately,
    // then time a steady-state loop - the number that matters per frame.
    t0 = std::chrono::steady_clock::now();
    cv::Mat mask = seg.inferMask(frame);
    std::printf("first inferMask: %ld ms\n", msSince(t0));

    constexpr int kIters = 50;
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i)
        mask = seg.inferMask(frame);
    std::printf("steady state:    %.1f ms/frame over %d iters\n",
                static_cast<double>(msSince(t0)) / kIters, kIters);

    const double dynamicPct =
        100.0 * static_cast<double>(cv::countNonZero(mask == 0)) / (mask.rows * mask.cols);
    std::printf("mask %dx%d, dynamic: %.1f%%\n", mask.cols, mask.rows, dynamicPct);

    cv::Mat overlay = frame.clone();
    overlay.setTo(cv::Scalar(0, 0, 210), mask == 0);
    cv::addWeighted(overlay, 0.5, frame, 0.5, 0.0, overlay);
    cv::imwrite("trt_mask.png", mask * 255);
    cv::imwrite("trt_overlay.png", overlay);
    std::printf("wrote trt_mask.png, trt_overlay.png\n");
    return 0;
}
