/**
 * Module: AllFeature-VSLAM - Segmentation-TensorRT - test_segmentation.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 0.2 (engine build/load lifecycle)
 * - Created: 2026-08-17
 * - License: GPLv3 License
 *
 * Standalone test harness for the online segmentation module (segmentation.md,
 * Phase 1). Step 2: build-or-load a TensorRT engine from an exported ONNX and
 * print its I/O tensors. Next step grows this into full inference, verified
 * against the Python self-check in convert2onnx/export_efficientvit_seg.py.
 *
 * Usage: test_segmentation <model.onnx> [fp16|fp32]
 */

#include <chrono>
#include <cstdio>

#include "tensorrt_seg.hpp"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s <model.onnx> [fp16|fp32]\n", argv[0]);
        return 1;
    }
    const std::string onnxPath = argv[1];
    const std::string precision = argc > 2 ? argv[2] : "fp16";

    const auto t0 = std::chrono::steady_clock::now();
    segmentation::TensorRTSeg seg(onnxPath, precision);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

    std::printf("engine %s in %ld ms: %s\nIO tensors:\n%s", seg.loadedFromCache() ? "loaded" : "built",
                ms, seg.enginePath().c_str(), seg.describeIOTensors().c_str());
    return 0;
}
