/**
 * Module: AllFeature-VSLAM - Segmentation-TensorRT - tensorrt_seg.hpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 0.3 (full inference: preprocess -> enqueueV3 -> binary mask)
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
 *
 * The sidecar '<onnx>.classes.yaml' (written by the export script) supplies
 * everything inference must know but never hardcode: normalization constants,
 * tensor names, and which class ids are dynamic.
 */

#ifndef SEGMENTATION_TENSORRT_TENSORRT_SEG_HPP
#define SEGMENTATION_TENSORRT_TENSORRT_SEG_HPP

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

namespace segmentation
{

class TensorRTSeg
{
  public:
    /**
     * Build or load the TensorRT engine for an exported segmentation ONNX and
     * prepare all inference resources (context, stream, GPU buffers).
     * @param onnxPath   model from export_efficientvit_seg.py
     * @param precision  "fp16" (default) or "fp32"; part of the cache filename,
     *                   so both engines can coexist next to one ONNX
     * @param classesYamlPath sidecar path; empty -> '<onnxPath>.classes.yaml'
     * @throws std::runtime_error on any unrecoverable setup failure
     */
    explicit TensorRTSeg(const std::string& onnxPath, const std::string& precision = "fp16",
                         const std::string& classesYamlPath = "");
    ~TensorRTSeg();

    TensorRTSeg(const TensorRTSeg&) = delete;
    TensorRTSeg& operator=(const TensorRTSeg&) = delete;

    /**
     * Segment one frame into the repo-wide mask convention: CV_8UC1, same size
     * as 'frame', 1 = static pixel, 0 = dynamic pixel. Grayscale input is
     * replicated to 3 channels (matching cv2.imread in the Python reference).
     * Synchronous: returns once the mask is ready (~few ms fp16).
     */
    cv::Mat inferMask(const cv::Mat& frame);

    /** Human-readable one-line description of every engine I/O tensor. */
    std::string describeIOTensors() const;

    /** True if the engine came from the cache file, false if it was built now. */
    bool loadedFromCache() const { return loadedFromCache_; }

    const std::string& enginePath() const { return enginePath_; }
    int inputWidth() const { return inputW_; }
    int inputHeight() const { return inputH_; }

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

    /** Read normalization, tensor names and dynamic class ids from the sidecar. */
    void loadClassesYaml(const std::string& yamlPath);

    /** Create execution context + CUDA stream, allocate and bind GPU buffers
     *  sized from the engine's own tensor shapes (not the YAML - the engine is
     *  the ground truth; the YAML is cross-checked against it). */
    void setupInference();

    std::string precision_;
    std::string enginePath_;
    bool loadedFromCache_{false};

    // --- sidecar-provided inference parameters ---
    std::array<float, 3> mean_{};
    std::array<float, 3> std_{};
    std::string inputName_{"input"};
    std::string outputName_{"class_map"};
    std::vector<uint8_t> staticLut_; // per class id: 1 = static, 0 = dynamic
    int dilateRadiusPx_{3};          // dynamic-region dilation at network resolution

    int inputW_{0};
    int inputH_{0};

    // Declaration order is destruction order in reverse: context before engine,
    // engine before runtime - each must die before the object that created it.
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_{nullptr};
    void* dInput_{nullptr};             // device: [1,3,H,W] float32
    void* dOutput_{nullptr};            // device: [1,H,W] int32
    std::vector<float> hostInput_;      // staging, CHW-planar
    std::vector<int32_t> hostOutput_;   // staging, class ids
};

} // namespace segmentation

#endif // SEGMENTATION_TENSORRT_TENSORRT_SEG_HPP
