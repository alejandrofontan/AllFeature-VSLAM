/**
 * Module: AllFeature-VSLAM - onnx/OrtSession.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.0
 * - Created: 2026-10-16
 * - License: GPLv3 License
 */

#include "onnx/OrtSession.h"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "afvslam_log.hpp"

namespace AF_VSLAM::onnx
{

namespace
{

/// ONNX Runtime's log channel, mapped onto the host's levels.
void ort_log(void* /*param*/, OrtLoggingLevel severity, const char* category, const char* /*logid*/,
             const char* /*code_location*/, const char* message)
{
    switch (severity)
    {
        case ORT_LOGGING_LEVEL_FATAL:
        case ORT_LOGGING_LEVEL_ERROR:
            AF_ERROR("[ORT] " << category << ": " << message);
            break;
        case ORT_LOGGING_LEVEL_WARNING:
            AF_WARN("[ORT] " << category << ": " << message);
            break;
        default:
            break;   // INFO / VERBOSE: the session logs what matters itself
    }
}

/// One environment per process (ONNX Runtime's own recommendation): every OrtSession shares it.
Ort::Env& shared_env()
{
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "allfeature", &ort_log, nullptr);
    return env;
}

bool has_provider(const std::vector<std::string>& available, const char* name)
{
    return std::find(available.begin(), available.end(), name) != available.end();
}

std::string shape_to_string(const std::vector<int64_t>& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
        oss << (i ? "," : "") << shape[i];
    oss << "]";
    return oss.str();
}

} // namespace

OrtSession::OrtSession(const std::string& onnx_path) : OrtSession(onnx_path, OrtSettings{}) {}

OrtSession::OrtSession(const std::string& onnx_path, const OrtSettings& settings)
    : model_path_(onnx_path), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    if (settings.threads < 1)
        throw std::runtime_error("OrtSession: threads must be >= 1");
    shared_env();   // must exist before any provider API call: the CUDA provider options use the Env's logger
    if (!std::filesystem::exists(model_path_))
        throw std::runtime_error("OrtSession: model file not found: " + model_path_ +
                                 " (export it with `pixi run -e export-features export-features` or run the "
                                 "download task)");
    // A fixed small thread count, never the core count: the models share the machine with the SLAM threads.
    // Sequential execution: one graph at a time, no inter-op parallelism to reason about.
    options_.SetIntraOpNumThreads(settings.threads);
    options_.SetInterOpNumThreads(1);
    options_.SetExecutionMode(ORT_SEQUENTIAL);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    provider_ = append_providers(options_, settings);
    try
    {
        session_ = std::make_unique<Ort::Session>(shared_env(), model_path_.c_str(), options_);
    }
    catch (const Ort::Exception& e)
    {
        if (provider_ == "cpu")
            throw std::runtime_error("OrtSession: session creation failed for " + model_path_ + ": " + e.what());
        // A provider that cannot take this graph (Core ML compiles only bounded shapes: a dynamic-axis model may
        // fail here rather than fall back node by node) must not take the SLAM down - the CPU provider always can.
        AF_WARN("OrtSession: provider '" << provider_ << "' cannot create a session for " << model_path_ << " ("
                << e.what() << ") - falling back to the CPU provider");
        options_ = Ort::SessionOptions();
        options_.SetIntraOpNumThreads(settings.threads);
        options_.SetInterOpNumThreads(1);
        options_.SetExecutionMode(ORT_SEQUENTIAL);
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        provider_ = "cpu";
        try
        {
            session_ = std::make_unique<Ort::Session>(shared_env(), model_path_.c_str(), options_);
        }
        catch (const Ort::Exception& e2)
        {
            throw std::runtime_error("OrtSession: session creation failed for " + model_path_ + ": " + e2.what());
        }
    }
    read_io_names();
}

OrtSession::~OrtSession() = default;

std::string OrtSession::append_providers(Ort::SessionOptions& options, const OrtSettings& settings) const
{
    const std::vector<std::string> available = Ort::GetAvailableProviders();
    const bool has_cuda = has_provider(available, "CUDAExecutionProvider");
    const bool has_coreml = has_provider(available, "CoreMLExecutionProvider");
    std::string chosen = settings.provider;
    if (chosen != "auto" && chosen != "cuda" && chosen != "coreml" && chosen != "cpu")
        throw std::runtime_error("OrtSession: provider must be auto|cuda|coreml|cpu, got " + settings.provider);
    if ((chosen == "cuda" && !has_cuda) || (chosen == "coreml" && !has_coreml))
    {
        std::ostringstream list;
        for (const auto& p : available)
            list << " " << p;
        AF_WARN("OrtSession: provider '" << chosen << "' is not available in this ONNX Runtime (" << list.str()
                << " ) - falling back to auto");
        chosen = "auto";
    }
    if (chosen == "auto")
        chosen = has_cuda ? "cuda" : has_coreml ? "coreml" : "cpu";
    if (chosen == "cuda")
    {
        // use_tf32=0: the CUDA provider runs float32 convolutions in TF32 by default on Ampere+ GPUs (10-bit
        // mantissa). Measured on SuperPoint (test_feature_parity, table_3): with TF32 the scores differ from the
        // CPU/PyTorch reference by 1e-3 and ~1 % of the NMS ties flip (99.2-100 % of the keypoints kept); without
        // it the output is bit-identical to the reference, for 9.3 instead of 8.5 ms per frame. Exactness wins.
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));
        try
        {
            const char* keys[] = {"device_id", "cudnn_conv_use_max_workspace", "use_tf32"};
            const char* values[] = {"0", "1", "0"};
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys, values, 3));
            options.AppendExecutionProvider_CUDA_V2(*cuda_options);
        }
        catch (...)
        {
            Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
            throw;
        }
        Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
    }
    else if (chosen == "coreml")
    {
        // MLProgram: the Core ML format with the widest operator coverage (macOS 12+). Dynamic input shapes are
        // allowed here (image size, keypoint counts change per call); the compiled model is cached next to the
        // ONNX so later sessions skip the compile. Whether dynamic shapes cost Core ML speed is a step-7 measurement.
        const std::filesystem::path cache_dir = std::filesystem::path(model_path_).parent_path() / "coreml_cache";
        std::unordered_map<std::string, std::string> coreml_options{
            {"ModelFormat", "MLProgram"},
            {"MLComputeUnits", settings.coreml_compute_units},
            {"RequireStaticInputShapes", "0"},
            {"EnableOnSubgraphs", "0"},
            {"ModelCacheDirectory", cache_dir.string()},
        };
        options.AppendExecutionProvider("CoreML", coreml_options);
    }
    // The CPU provider is implicit and last: it takes every node the others decline.
    return chosen;
}

void OrtSession::read_io_names()
{
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session_->GetInputCount(); ++i)
        input_names_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session_->GetOutputCount(); ++i)
        output_names_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
}

std::vector<Tensor> OrtSession::run(const std::vector<TensorView>& inputs)
{
    if (inputs.size() != input_names_.size())
        throw std::runtime_error("OrtSession: " + model_path_ + " expects " + std::to_string(input_names_.size()) +
                                 " inputs, got " + std::to_string(inputs.size()) + "\n" + describe_io());
    // The input Ort::Values wrap the caller's buffers (no copy); the graph's declared input order is used.
    std::vector<const char*> in_names, out_names;
    std::vector<Ort::Value> in_values;
    in_names.reserve(inputs.size());
    in_values.reserve(inputs.size());
    for (const std::string& name : input_names_)
    {
        const auto it = std::find_if(inputs.begin(), inputs.end(), [&](const TensorView& t) { return t.name == name; });
        if (it == inputs.end())
            throw std::runtime_error("OrtSession: input '" + name + "' not provided for " + model_path_);
        const size_t numel = static_cast<size_t>(
            std::accumulate(it->shape.begin(), it->shape.end(), int64_t{1}, std::multiplies<int64_t>()));
        in_names.push_back(name.c_str());
        in_values.push_back(Ort::Value::CreateTensor<float>(memory_info_, const_cast<float*>(it->data), numel,
                                                            it->shape.data(), it->shape.size()));
    }
    for (const std::string& name : output_names_)
        out_names.push_back(name.c_str());

    std::vector<Ort::Value> out_values;
    try
    {
        out_values = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), in_values.data(), in_values.size(),
                                   out_names.data(), out_names.size());
    }
    catch (const Ort::Exception& e)
    {
        throw std::runtime_error("OrtSession: Run failed for " + model_path_ + ": " + e.what());
    }

    std::vector<Tensor> outputs(out_values.size());
    for (size_t i = 0; i < out_values.size(); ++i)
    {
        const auto info = out_values[i].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("OrtSession: output '" + output_names_[i] + "' of " + model_path_ +
                                     " is not float32 - wrong export?");
        outputs[i].name = output_names_[i];
        outputs[i].shape = info.GetShape();
        const float* data = out_values[i].GetTensorData<float>();
        outputs[i].data.assign(data, data + info.GetElementCount());
    }
    return outputs;
}

std::string OrtSession::describe_io() const
{
    std::ostringstream oss;
    for (size_t i = 0; i < session_->GetInputCount(); ++i)
    {
        const Ort::TypeInfo type = session_->GetInputTypeInfo(i);   // the view below must not outlive it
        const auto info = type.GetTensorTypeAndShapeInfo();
        oss << "  input  " << input_names_[i] << " " << shape_to_string(info.GetShape())
            << " dtype=" << static_cast<int>(info.GetElementType()) << "\n";
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i)
    {
        const Ort::TypeInfo type = session_->GetOutputTypeInfo(i);
        const auto info = type.GetTensorTypeAndShapeInfo();
        oss << "  output " << output_names_[i] << " " << shape_to_string(info.GetShape())
            << " dtype=" << static_cast<int>(info.GetElementType()) << "\n";
    }
    return oss.str();
}

} // namespace AF_VSLAM::onnx
