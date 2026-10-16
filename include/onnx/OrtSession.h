/**
 * Module: AllFeature-VSLAM - onnx/OrtSession.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 *
 * OrtSession: one ONNX Runtime session for one exported model file, shared by the learned feature models
 * (SuperPoint, ALIKED, the LightGlue matchers; docs/plans/learned_features_onnxruntime.md step 4). The same
 * design as placecell's MegaLoc backend (Thirdparty/placecell/src/megaloc/onnxruntime_megaloc.cpp, phase 2),
 * generalised to graphs whose tensor shapes change on every call - an image of any size in, a data-dependent
 * number of keypoints out - so inputs are passed with their shape per call and output shapes are read back.
 *
 * One C++ path on every platform; the hardware is an execution-provider choice made at session creation:
 *
 *   provider  "auto"   CUDA where ONNX Runtime was built with it, else CoreML (macOS), else CPU
 *             "cuda"   NVIDIA GPU (the CUDA execution provider; cuDNN at runtime)
 *             "coreml" Apple GPU / Neural Engine through Core ML (MLProgram, compiled once per model file and
 *                      cached in <model dir>/coreml_cache)
 *             "cpu"    the CPU provider only (`threads` intra-op threads)
 *
 * A requested provider that is not available falls back to "auto" with a warning; the CPU provider is always
 * last, so any node a provider does not support still runs. Precision is a property of the file (the export
 * writes '<model>.onnx' in fp32 and '<model>_fp16.onnx' with fp16 weights; I/O float32 in both) and is read by
 * the model wrappers from the sidecar '<model>.onnx.yaml'.
 *
 * Threading: Ort::Session::Run may be called concurrently (ONNX Runtime guarantee); run() keeps no per-call state
 * in the object, so one OrtSession can serve the tracking and the local-mapping threads at once.
 */

#ifndef AF_VSLAM_ONNX_ORT_SESSION_H
#define AF_VSLAM_ONNX_ORT_SESSION_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace AF_VSLAM::onnx
{

/// The runtime choices the settings YAML makes for every learned feature model (features_provider,
/// features_threads, features_coreml_units; plan Q1).
struct OrtSettings
{
    std::string provider{"auto"};
    int threads{4};
    std::string coreml_compute_units{"ALL"};   // "ALL" | "CPUAndGPU" | "CPUAndNeuralEngine" | "CPUOnly"
};

/// A float32 input tensor for one run(): a view of host memory the caller keeps alive during the call.
struct TensorView
{
    std::string name;
    const float* data{nullptr};
    std::vector<int64_t> shape;
};

/// A float32 output tensor of one run(): the session's buffer copied out, with the shape read back.
struct Tensor
{
    std::string name;
    std::vector<float> data;
    std::vector<int64_t> shape;

    [[nodiscard]] int64_t dim(size_t i) const { return shape.at(i); }
};

class OrtSession
{
  public:
    /// Create the session for `onnx_path`. @throws std::runtime_error on any unrecoverable setup failure.
    explicit OrtSession(const std::string& onnx_path);
    OrtSession(const std::string& onnx_path, const OrtSettings& settings);
    ~OrtSession();
    OrtSession(const OrtSession&) = delete;
    OrtSession& operator=(const OrtSession&) = delete;

    /// Run the graph once on `inputs` (every declared input, any order) and return every declared output.
    /// Synchronous. @throws std::runtime_error if a name is unknown or ONNX Runtime fails.
    std::vector<Tensor> run(const std::vector<TensorView>& inputs);

    /// The provider the session got: "cuda", "coreml" or "cpu".
    [[nodiscard]] const std::string& provider() const { return provider_; }
    [[nodiscard]] const std::string& model_path() const { return model_path_; }
    [[nodiscard]] const std::vector<std::string>& input_names() const { return input_names_; }
    [[nodiscard]] const std::vector<std::string>& output_names() const { return output_names_; }
    /// One line per model input/output: name, shape (symbolic dims as -1), element type.
    [[nodiscard]] std::string describe_io() const;

  private:
    std::string append_providers(Ort::SessionOptions& options, const OrtSettings& settings) const;
    void read_io_names();

    std::string model_path_;
    std::string provider_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace AF_VSLAM::onnx

#endif // AF_VSLAM_ONNX_ORT_SESSION_H
