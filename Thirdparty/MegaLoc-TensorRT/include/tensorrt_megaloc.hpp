/**
 * Module: AllFeature-VSLAM - MegaLoc-TensorRT - tensorrt_megaloc.hpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-08-28
 * - License: GPLv3 License (MegaLoc itself is MIT)
 *
 * TensorRTMegaLoc: image -> MegaLoc global descriptor (8448-d, L2-normalised) for
 * the online visual-place-recognition backend of AllFeature-VSLAM (`vpr: megaloc`).
 * Wraps the full TensorRT lifecycle for the ONNX produced by
 * convert2onnx/export_megaloc.py (input [1,3,H,W] float32 -> descriptor [1,D]).
 *
 * Same engine-cache policy as Segmentation-TensorRT: building an engine benchmarks
 * kernels for every layer on THIS GPU (~1-2 min for a ViT-B); the blob is serialized
 * to '<onnx>.<precision>.engine' next to the ONNX and deserialized in milliseconds on
 * later runs. A blob is tied to GPU model + TensorRT version, so a failed load falls
 * back to a rebuild that overwrites the cache.
 *
 * The sidecar '<onnx>.yaml' (written by the export script) supplies what inference
 * must know but never hardcode: input size, normalisation, tensor names, descriptor
 * dimension.
 */

#ifndef MEGALOC_TENSORRT_TENSORRT_MEGALOC_HPP
#define MEGALOC_TENSORRT_TENSORRT_MEGALOC_HPP

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

namespace megaloc
{

class TensorRTMegaLoc
{
  public:
    /**
     * Build or load the TensorRT engine for an exported MegaLoc ONNX and prepare all
     * inference resources (context, stream, GPU buffers).
     * @param onnxPath   model from export_megaloc.py
     * @param precision  "fp16" (default: fp16 ViT backbone, fp32 aggregator - see
     *                   buildEngine) or "fp32"; part of the cache filename, so both
     *                   engines can coexist next to one ONNX
     * @param sidecarYamlPath sidecar path; empty -> '<onnxPath>.yaml'
     * @throws std::runtime_error on any unrecoverable setup failure
     */
    explicit TensorRTMegaLoc(const std::string& onnxPath, const std::string& precision = "fp16",
                             const std::string& sidecarYamlPath = "");
    ~TensorRTMegaLoc();

    TensorRTMegaLoc(const TensorRTMegaLoc&) = delete;
    TensorRTMegaLoc& operator=(const TensorRTMegaLoc&) = delete;

    /**
     * Global descriptor of one image, L2-normalised (re-normalised on the host so fp16
     * rounding cannot leave |d| != 1). Input in OpenCV's native BGR order (cv::imread);
     * grayscale is replicated to 3 channels, BGRA drops alpha. Synchronous: returns once
     * the descriptor is ready (~5 ms fp16 at 322x322 on an RTX 4000 Ada).
     * NOT thread-safe: one execution context, callers serialise externally.
     */
    std::vector<float> infer(const cv::Mat& imageBGR);

    /** Cosine similarity of two descriptors (dot product for unit vectors). */
    static float cosine(const std::vector<float>& a, const std::vector<float>& b);

    /** Human-readable one-line description of every engine I/O tensor. */
    std::string describeIOTensors() const;

    /** True if the engine came from the cache file, false if it was built now. */
    bool loadedFromCache() const { return loadedFromCache_; }

    const std::string& enginePath() const { return enginePath_; }
    int descriptorDim() const { return descriptorDim_; }
    int inputWidth() const { return inputW_; }
    int inputHeight() const { return inputH_; }

  private:
    /** TensorRT's only output channel; lower the filter to kINFO when debugging a build. */
    class Logger final : public nvinfer1::ILogger
    {
      public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    /** ONNX -> INetworkDefinition -> fp16/fp32 build -> blob written to enginePath_. */
    void buildEngine(const std::string& onnxPath);

    /** Deserialize enginePath_; false (not a throw) so the constructor can rebuild -
     *  also when the cache predates the ONNX it was built from. */
    bool tryLoadEngine(const std::string& onnxPath);

    /** Read normalisation, tensor names, input size and descriptor dim from the sidecar. */
    void loadSidecarYaml(const std::string& yamlPath);

    /** Create execution context + CUDA stream, allocate and bind GPU buffers sized from
     *  the engine's own tensor shapes (the YAML is cross-checked against them). */
    void setupInference();

    std::string precision_;
    std::string enginePath_;
    bool loadedFromCache_{false};

    // --- sidecar-provided inference parameters ---
    std::array<float, 3> mean_{};
    std::array<float, 3> std_{};
    std::string inputName_{"input"};
    std::string outputName_{"descriptor"};
    int inputW_{0};
    int inputH_{0};
    int descriptorDim_{0};

    // Declaration order is destruction order in reverse: context before engine,
    // engine before runtime.
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_{nullptr};
    void* dInput_{nullptr};        // device: [1,3,H,W] float32
    void* dOutput_{nullptr};       // device: [1,D] float32
    std::vector<float> hostInput_; // staging, CHW-planar
    std::vector<float> hostOutput_;
};

} // namespace megaloc

#endif // MEGALOC_TENSORRT_TENSORRT_MEGALOC_HPP
