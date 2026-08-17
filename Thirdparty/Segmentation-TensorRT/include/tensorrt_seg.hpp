/**
 * Module: AllFeature-VSLAM - Segmentation-TensorRT - tensorrt_seg.hpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 0.2 (engine build/load lifecycle)
 * - Created: 2026-08-17
 * - License: GPLv3 License
 *
 * TensorRTSeg: online static/dynamic segmentation for AllFeature-VSLAM
 * (segmentation.md, Phase 1). Wraps the full TensorRT lifecycle for the
 * ONNX models produced by convert2onnx/export_efficientvit_seg.py
 * (input [1,3,H,W] float32 -> class_map [1,H,W] int32).
 *
 * Engine caching: building an engine means TensorRT benchmarks kernel
 * implementations for every layer ON THIS GPU (~a minute); the result is
 * serialized to '<onnx>.<precision>.engine' next to the ONNX. Later runs
 * deserialize that file in milliseconds. An engine blob is tied to the GPU
 * model and TensorRT version, so the policy is: try to load, and on any
 * failure rebuild and overwrite the cache.
 */

#ifndef SEGMENTATION_TENSORRT_TENSORRT_SEG_HPP
#define SEGMENTATION_TENSORRT_TENSORRT_SEG_HPP

#include <memory>
#include <string>

#include <NvInfer.h>

namespace segmentation
{

class TensorRTSeg
{
  public:
    /**
     * Build or load the TensorRT engine for an exported segmentation ONNX.
     * @param onnxPath  model from export_efficientvit_seg.py
     * @param precision "fp16" (default) or "fp32"; part of the cache filename,
     *                  so both engines can coexist next to one ONNX
     * @throws std::runtime_error if neither loading nor building succeeds
     */
    explicit TensorRTSeg(const std::string& onnxPath, const std::string& precision = "fp16");

    /** Human-readable one-line description of every engine I/O tensor. */
    std::string describeIOTensors() const;

    /** True if the engine came from the cache file, false if it was built now. */
    bool loadedFromCache() const { return loadedFromCache_; }

    const std::string& enginePath() const { return enginePath_; }

  private:
    /**
     * Everything TensorRT prints (build diagnostics, kernel selection notes,
     * deserialization errors) arrives through this callback - it is TRT's only
     * output channel. Severity is ordered ERROR < WARNING < INFO < VERBOSE;
     * lower the filter to kINFO when debugging an engine build.
     */
    class Logger final : public nvinfer1::ILogger
    {
      public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    /** ONNX -> INetworkDefinition -> fp16/fp32 build -> blob written to enginePath_. */
    void buildEngine(const std::string& onnxPath);

    /** Deserialize enginePath_; returns false (instead of throwing) so the
     *  constructor can fall back to a rebuild on stale/foreign caches. */
    bool tryLoadEngine();

    std::string precision_;
    std::string enginePath_;
    bool loadedFromCache_{false};

    // Declaration order is destruction order in reverse: the engine must be
    // destroyed before the runtime that deserialized it, so runtime_ first.
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
};

} // namespace segmentation

#endif // SEGMENTATION_TENSORRT_TENSORRT_SEG_HPP
