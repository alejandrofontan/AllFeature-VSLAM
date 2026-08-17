/**
 * Module: AllFeature-VSLAM - Segmentation-TensorRT - test_segmentation.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 0.1 ("hello TensorRT")
 * - Created: 2026-08-17
 * - License: GPLv3 License
 *
 * Standalone test harness for the online segmentation module (segmentation.md,
 * Phase 1). Step 1: prove TensorRT links and initializes - print the library
 * version and create/destroy a builder. Later steps grow this into the full
 * ONNX -> engine -> inference -> mask pipeline, verified against the Python
 * self-check in convert2onnx/export_efficientvit_seg.py.
 */

#include <cstdio>
#include <memory>

#include <NvInfer.h>

/**
 * Every TensorRT entry point takes an ILogger - it is the ONLY channel through
 * which TRT reports diagnostics (e.g. why an engine build failed). Severity is
 * ordered ERROR < WARNING < INFO < VERBOSE, so "<= kWARNING" means "warnings
 * and worse". Lower the filter to kINFO when debugging engine builds.
 */
class Logger final : public nvinfer1::ILogger
{
  public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::printf("[TensorRT] %s\n", msg);
    }
};

int main()
{
    // getInferLibVersion() returns major*10000 + minor*100 + patch.
    const int32_t v = getInferLibVersion();
    std::printf("TensorRT %d.%d.%d (raw %d)\n", v / 10000, (v % 10000) / 100, v % 100, v);

    Logger logger;
    // TRT factory functions return raw pointers the caller owns; since TRT 8+
    // their destructors are public, so a plain unique_ptr is the idiom.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        std::printf("createInferBuilder failed\n");
        return 1;
    }
    std::printf("builder created OK (platform has fast fp16: %s)\n",
                builder->platformHasFastFp16() ? "yes" : "no");
    return 0;
}
