/**
 * Module: AllFeature-VSLAM - onnx/OrtSession.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 *
 * OrtSession: one ONNX Runtime session for one exported model file, shared by the learned feature models
 * (SuperPoint, ALIKED, the LightGlue matchers). The same
 * design as placecell's MegaLoc backend (Thirdparty/placecell/src/megaloc/onnxruntime_megaloc.cpp),
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
 * writes '<model>_fp32.onnx' and '<model>_fp16.onnx' with fp16 weights; I/O float32 in both) and is read by
 * the model wrappers from the sidecar '<model>.onnx.yaml'.
 *
 * Threading: run() keeps no per-call state, but the calls on one session are SERIALISED by a mutex. ONNX Runtime
 * documents Ort::Session::Run as safe to call concurrently; with the CUDA provider (ORT 1.31) two threads running the
 * LightGlue session at once (Tracking's reference-keyframe match + LocalMapping's five parallel neighbour matches)
 * returned zero matches to one of them on almost every keyframe (2026-10-16). The GPU runs the
 * kernels one after the other anyway, so the lock costs nothing there; it only removes the overlap of the host-side
 * packing, which is microseconds. One OrtSession still serves the tracking and the local-mapping threads; they
 * take turns.
 *
 * The turns are not first-come-first-served: a thread that called set_run_priority(true) goes ahead of every waiting
 * thread that did not. Tracking is the priority thread. Local Mapping matches a new keyframe against several
 * neighbours at once, so its Runs queue up on the shared LightGlue session, and the next frame's match used to wait
 * behind all of them (four ALIKED matches, about 80 ms, on the first frame after every keyframe). With priority it
 * waits for at most the one Run already in progress, which cannot be interrupted.
 */

#ifndef AF_VSLAM_ONNX_ORT_SESSION_H
#define AF_VSLAM_ONNX_ORT_SESSION_H

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace AF_VSLAM::onnx
{

/// The runtime choices the settings YAML makes for every learned feature model (features_provider,
/// features_threads, features_coreml_units).
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

/// The provider a session asked for `requested` ("auto" | "cuda" | "coreml" | "cpu") gets in this ONNX Runtime build:
/// "cuda", "coreml" or "cpu". An unavailable one falls back to auto (with a warning when `warn`).
/// @throws std::runtime_error if `requested` is none of the four
std::string resolve_provider(const std::string& requested, bool warn = true);

/// Marks the calling thread's run() calls, on every session, as priority (true) or normal (false, the default):
/// a waiting priority call is served before any waiting normal call. See the header comment, Threading.
void set_run_priority(bool priority);

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
    // One Run at a time per session, priority callers first (see the header comment, Threading).
    std::mutex run_mutex_;
    std::condition_variable run_turn_;
    bool running_{false};          // a Run is in progress
    int priority_waiting_{0};      // priority callers waiting for their turn
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace AF_VSLAM::onnx

#endif // AF_VSLAM_ONNX_ORT_SESSION_H
